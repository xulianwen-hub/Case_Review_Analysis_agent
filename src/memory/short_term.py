"""
src.memory.short_term —— 短期工作记忆（会话级）

💡 面试讲解点
    · 是什么：单次对话生命周期内的滑动窗口 Buffer，保存最近 K 轮对话 + 本轮 RAG 检索结果缓存
    · 为什么用滑动窗口：LLM 上下文窗口有限（DeepSeek 是 128K，但不是无限），不能把所有历史拼进去
    · 设计权衡：
        ✔ 选 list + 截断（简单、O(N) 对 20-50 条对话完全足够）
        ✘ 没选 deque：因为需要支持「只读 user 消息」这种任意过滤读，deque 不如 list 灵活
    · RAG 结果缓存：同一轮多子问题共享检索结果，不用重复查 Milvus/BM25（省 300-800ms/轮）
"""
import time
from typing import Optional
from dataclasses import dataclass, field

from .base import BaseMemory, MemoryItem


# ============================================================
# 配置常量（可以单独拆到 config.py，这里集中写方便面试解释）
# ============================================================
# 滑动窗口：最多保留最近 N 轮「user-assistant 对话对」（约等于 2*N 条消息）
DEFAULT_MAX_DIALOGUE_PAIRS = 20
# RAG 结果缓存：单轮对话的检索结果，进入下一轮时自动清理
RAG_CACHE_KEY = "__rag_cache__"
TOOL_CACHE_KEY = "__tool_cache__"


class ShortTermMemory(BaseMemory):
    """
    短期工作记忆
        - 最近 K 轮对话（滑动窗口）
        - 单轮 RAG 检索结果缓存
        - 单轮 Tool 调用结果缓存
    """

    def __init__(self, max_dialogue_pairs: int = DEFAULT_MAX_DIALOGUE_PAIRS):
        self.max_dialogue_pairs = max_dialogue_pairs
        self._buffer: list[MemoryItem] = []
        self._monotonic_ts: int = 0

    # ============================================================
    # BaseMemory 接口实现
    # ============================================================
    def write(self, role: str, content, **metadata) -> MemoryItem:
        self._monotonic_ts += 1
        item = MemoryItem(
            role=role,
            content=content,
            timestamp=self._monotonic_ts,
            metadata=metadata,
            tokens_estimate=metadata.pop("tokens_estimate", self._estimate_tokens(content)),
        )
        self._buffer.append(item)

        # 滑动窗口裁剪：只对 user/assistant 两类消息计入窗口计数
        # （不裁剪 slot_update / rag_result，它们数量很少且 Orchestrator 读的时候自己会判断 role）
        self._trim_dialogue_window()
        return item

    def read(self, limit: Optional[int] = None, **filters) -> list[MemoryItem]:
        result = self._buffer
        # 过滤：支持 role= / metadata 字段匹配
        if filters:
            def _match(it: MemoryItem) -> bool:
                for k, v in filters.items():
                    if k == "role":
                        if it.role != v:
                            return False
                        continue  # role 检查通过，跳过 metadata 检查（metadata 不会有 role 这个 key）
                    if it.metadata.get(k) != v:
                        return False
                return True
            result = [it for it in result if _match(it)]
        if limit is not None and len(result) > limit:
            result = result[-limit:]
        return list(result)  # 返回复制，避免外部修改 buffer

    def clear(self) -> None:
        self._buffer.clear()
        self._monotonic_ts = 0

    # ============================================================
    # RAG / Tool 缓存（单轮）
    # ============================================================
    def cache_rag_result(self, tag: str, payload) -> None:
        """把检索结果记下来，metadata 标 type=RAG_CACHE_KEY，下一轮 read_rag_cache 能取"""
        self.write(
            role="rag_result",
            content=payload,
            **{RAG_CACHE_KEY: tag},
        )

    def read_rag_cache(self, tag: Optional[str] = None) -> Optional[MemoryItem]:
        """取最新一次 RAG 结果（tag 可过滤，比如只要 laws / cases）"""
        for it in reversed(self._buffer):
            if it.role != "rag_result":
                continue
            if tag is None or it.metadata.get(RAG_CACHE_KEY) == tag:
                return it
        return None

    def cache_tool_result(self, tool_name: str, payload) -> None:
        self.write(role="tool_result", content=payload, **{TOOL_CACHE_KEY: tool_name})

    def read_tool_cache(self, tool_name: str) -> Optional[MemoryItem]:
        for it in reversed(self._buffer):
            if it.role == "tool_result" and it.metadata.get(TOOL_CACHE_KEY) == tool_name:
                return it
        return None

    # ============================================================
    # 上下文格式化（拼进 LLM Prompt 用的）
    # ============================================================
    def format_for_llm(self, max_turns: Optional[int] = None,
                       include_roles=("user", "assistant")) -> list[dict]:
        """
        把短期记忆格式化为 OpenAI 风格的 messages list：
            [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
        """
        turns_only = [it for it in self._buffer if it.role in include_roles]
        if max_turns is not None:
            turns_only = turns_only[-(max_turns * 2):]
        return [{"role": it.role, "content": str(it.content)} for it in turns_only]

    # ============================================================
    # 内部
    # ============================================================
    def _trim_dialogue_window(self) -> None:
        dialogue_items = [i for i in self._buffer if i.role in ("user", "assistant")]
        # 每一轮对话约等于 1 条 user + 1 条 assistant（共 2 条），用「成对计数」裁剪
        pairs = len(dialogue_items) // 2
        overflow_pairs = pairs - self.max_dialogue_pairs
        if overflow_pairs > 0:
            # 从缓冲区最旧的位置，删除 2*overflow_pairs 条对话消息
            remove_ids = {it.item_id for it in dialogue_items[: overflow_pairs * 2]}
            self._buffer = [it for it in self._buffer if it.item_id not in remove_ids]

    @staticmethod
    def _estimate_tokens(content) -> int:
        """极简 token 预估（中文 1.5 字/token，英文 4 字符/token）——够用来做压缩预算"""
        if not isinstance(content, str):
            try:
                import json
                text = json.dumps(content, ensure_ascii=False)
            except Exception:
                text = str(content)
        else:
            text = content
        # 粗略区分中英：ASCII 字符当英文，其余当中文
        en_chars = sum(1 for c in text if ord(c) < 128)
        cn_chars = len(text) - en_chars
        return max(1, int(cn_chars / 1.5 + en_chars / 4))