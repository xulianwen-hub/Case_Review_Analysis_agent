"""
src.memory.summary_buffer —— 对话历史渐进式压缩（解决 LLM Context Window 爆炸问题）

💡 面试讲解点（100% 会问：对话太长了超出上下文窗口怎么办？）
    · 问题背景：LLM 虽然有 128K / 200K 窗口，但如果是**长期会话（用户连续 50+ 轮咨询）**，
                每轮都拼接完整历史 → 成本飙升（token 数 × LLM 调用单价 = 真金白银）
                且模型对超长上下文的"中段遗忘"问题（Lost in the Middle 论文）
    · 常见方案对比，我们选哪个？
        ✘ 纯滑动窗口：删得太暴力，20 轮之前的「关键信息」（比如用户入职日期）可能被丢掉
        ✘ 每一轮都重新摘要：LLM 调用成本太高，没必要
        ✔ **渐进式滚动摘要（我们这个实现）**：
              · 设一个「压缩阈值」（比如 50 轮对话 or 3000 tokens）
              · 超过阈值后：把最旧的 N 轮 → 用 LLM 压缩成一段持久化总结
              · 新的对话继续堆滑动窗口
              · 读的时候：「持久化摘要 + 最近 K 轮原始对话」拼接给 LLM
              → 成本可控 + 关键信息不丢
    · 未来可扩展：摘要也分桶（按天/按主题），检索摘要时再做向量相似度 —— 就是
      所谓的「Memory Transformer / LongMem」思路，V1 先做滚动摘要够面试讲了
"""
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

from .base import BaseMemory, MemoryItem
from .short_term import ShortTermMemory


# ============================================================
# 默认阈值（调参可改）
# ============================================================
DEFAULT_ROLLING_TOKEN_THRESHOLD = 2000   # 超过这么多 tokens 就触发一次压缩
DEFAULT_ROLLING_MESSAGE_THRESHOLD = 12   # 或超过这么多条消息就触发压缩
DEFAULT_COMPRESS_RATIO = 0.5             # 每次压缩最旧的 50% 对话


# ============================================================
# Summary Buffer：短期记忆 + 滚动持久化摘要
# ============================================================
class SummaryBuffer(BaseMemory):
    """
    结构：
        [Persistent Summaries（LLM 压缩好的长文本）]
                 +
        [Recent Dialogue Window（最近的原始对话，短期记忆保存）]
    """

    def __init__(self,
                 token_threshold: int = DEFAULT_ROLLING_TOKEN_THRESHOLD,
                 message_threshold: int = DEFAULT_ROLLING_MESSAGE_THRESHOLD,
                 compress_ratio: float = DEFAULT_COMPRESS_RATIO):
        self.token_threshold = token_threshold
        self.message_threshold = message_threshold
        self.compress_ratio = compress_ratio
        # 滚动摘要：越靠后越新，每一次压缩追加一段
        self._summaries: list[MemoryItem] = []
        # 最近 N 轮原始对话
        self._recent = ShortTermMemory(max_dialogue_pairs=9999)  # 我们自己管裁剪
        self._compress_counter: int = 0

    # ============================================================
    # BaseMemory 接口实现
    # ============================================================
    def write(self, role: str, content, **metadata) -> MemoryItem:
        item = self._recent.write(role, content, **metadata)
        # 每次写完检查是否需要压缩
        # 注意：不在这里同步调 LLM（避免 write 变成异步慢操作），
        # 调用方（Orchestrator）自己在合适时机调用 maybe_compress()
        return item

    def read(self, limit: Optional[int] = None, **filters) -> list[MemoryItem]:
        combined: list[MemoryItem] = []
        if not filters:  # 过滤模式下只读 recent 部分（summaries 是摘要不适合 role 过滤）
            combined.extend(self._summaries)
        combined.extend(self._recent.read())
        if limit is not None and len(combined) > limit:
            combined = combined[-limit:]
        return combined

    def clear(self) -> None:
        self._summaries.clear()
        self._recent.clear()
        self._compress_counter = 0

    # ============================================================
    # 压缩控制（面试关键：这个函数单独能讲 3 分钟）
    # ============================================================
    def should_compress(self) -> bool:
        """判断是否到达阈值，需要触发压缩"""
        recent_items = self._recent.read()
        dialogue_items = [it for it in recent_items if it.role in ("user", "assistant")]
        dialogue_count = len(dialogue_items)
        dialogue_tokens = sum(it.tokens_estimate for it in dialogue_items)
        return (dialogue_count >= self.message_threshold
                or dialogue_tokens >= self.token_threshold)

    def maybe_compress(self,
                       llm_summarizer: Callable[[list[dict]], str],
                       force: bool = False) -> Optional[Tuple[int, int]]:
        """
        触发一次压缩：
            1. 挑出「要压缩的旧对话」（最近窗口的头部 compress_ratio 部分）
            2. 调 llm_summarizer(openai 风格 messages) → 得到摘要字符串
            3. 把摘要作为一条 MemoryItem 存进 _summaries，删除对应的旧消息

        返回 (压缩掉的消息数, 节省的 token 数)，没压缩返回 None
        """
        if not (force or self.should_compress()):
            return None

        all_recent = self._recent.read()
        # 只压缩 user/assistant 对话，保留 tool_result/rag_result（它们不占多少且是精确结构化数据）
        user_assi = [it for it in all_recent if it.role in ("user", "assistant")]
        if len(user_assi) < 4:
            return None  # 少于 2 轮对话不值得压缩

        n_to_compress = max(4, int(len(user_assi) * self.compress_ratio))
        n_to_compress = (n_to_compress // 2) * 2  # 对齐成偶数，按轮压缩
        to_compress = user_assi[:n_to_compress]

        old_tokens = sum(it.tokens_estimate for it in to_compress)

        # 调 LLM 压缩
        messages = [{"role": it.role, "content": str(it.content)} for it in to_compress]
        try:
            summary_text = llm_summarizer(messages)
        except Exception:
            # LLM 压缩失败 → 降级：把这些对话拼成一个纯文本地摘要，不丢数据
            summary_text = "【降级摘要】\n" + "\n".join(
                f"[{it.role}] {it.content}" for it in to_compress
            )

        # 追加到持久化摘要列表
        self._compress_counter += 1
        self._summaries.append(MemoryItem(
            role="system_summary",
            content=summary_text,
            metadata={"compressed_round": self._compress_counter,
                      "compressed_items": n_to_compress},
            tokens_estimate=max(1, len(summary_text) // 2),
        ))

        # 从 recent 中移除掉被压缩的条目（用 id 精确删除，不会误删中间夹杂的 tool_result）
        removed_ids = {it.item_id for it in to_compress}
        self._recent._buffer = [it for it in self._recent._buffer
                                if it.item_id not in removed_ids]

        saved_tokens = old_tokens - self._summaries[-1].tokens_estimate
        return n_to_compress, max(0, saved_tokens)

    # ============================================================
    # 输出给 LLM 的最终上下文格式
    # ============================================================
    def build_context_for_llm(self, max_recent_turns: int = 8) -> list[dict]:
        """
        返回：
            [
              {"role":"system","content":"【历史摘要 1】..."},
              {"role":"system","content":"【历史摘要 2】..."},
              {"role":"user", "...第 N-7 轮原始对话..."},
              ...
              {"role":"assistant", "...最近一轮回复..."}
            ]
        """
        result: list[dict] = []
        # 先放摘要
        for s in self._summaries:
            result.append({
                "role": "system",
                "content": f"【历史对话摘要 #{s.metadata.get('compressed_round', '?')}】\n{s.content}",
            })
        # 再放最近 N 轮原始对话
        recent_raw = self._recent.format_for_llm(max_turns=max_recent_turns)
        result.extend(recent_raw)
        return result