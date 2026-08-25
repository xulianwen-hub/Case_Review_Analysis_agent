"""
src.state.agent_prompt —— Agent 决策 System Prompt 构建器

把 Session 当前状态（FSM 状态 + 证据槽 + 可用工具 + 对话历史）动态组装成
LLM 能理解的上下文，让 LLM 自主决定下一步行动。

与 V1（规则驱动）的关键区别：
    V1：Orchestrator 硬编码 if/elif 判断下一步
    V2：LLM 看这段 Prompt 后自主决定下一步

输出格式约定（LLM 必须遵守）：
    每次回复必须包含一个 JSON 决策块，格式为：
    ```json
    {
      "action": "function_call" | "ask_user" | "final_answer",
      "reasoning": "一句话解释为什么做这个决策",
      ...
    }
    ```
    三种 action 的完整字段见下方 _DECISION_FORMAT。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import AgentSession


# ============================================================
# 决策输出格式（LLM 必须遵守）
# ============================================================
_DECISION_FORMAT = """
## 输出格式（严格遵守）

每次回复必须包含一个 JSON 决策块，放在 ```json 代码块中：

### 情况 A：需要调用工具
```json
{
  "action": "function_call",
  "tool_name": "工具名称",
  "tool_params": {"参数名": "参数值"},
  "reasoning": "为什么需要调这个工具"
}
```

### 情况 B：需要追问用户
```json
{
  "action": "ask_user",
  "message": "给用户看的自然语言追问内容",
  "reasoning": "为什么需要追问"
}
```

### 情况 C：给出最终结论
```json
{
  "action": "final_answer",
  "message": "给用户看的完整分析报告",
  "reasoning": "为什么可以给出最终结论"
}
```

在 JSON 块之外，你可以加一些自然语言（比如先安抚用户、解释你在做什么），
但 JSON 决策块是必须的，它决定了系统下一步的行为。
"""


# ============================================================
# 角色定义 + 工作流程（System Prompt 主体）
# ============================================================
_SYSTEM_PROMPT_TEMPLATE = """你是一个劳动法咨询 Agent，帮助遭遇劳动纠纷的普通劳动者分析案情、估算赔偿、给出行动建议。

## 你的身份
- 专业劳动法咨询专家，但用通俗易懂的大白话解释
- 提供的分析仅供参考，不构成法律意见
- 只能在用户明确要求时才生成法律文书

## 当前状态
{current_state_description}

## 已收集的案情信息
{evidence_slots_summary}

## 可用工具
{tools_description}

## 决策规则
1. **信息收集阶段（INIT / UNDERSTANDING / COLLECTING）**：
   - 如果 P0 优先级的证据槽（城市、入职日期、离职日期、是否签合同、月薪、解除原因）缺失超过 3 项
     → action=ask_user，每次最多问 2 个问题
   - 如果 P0+P1 证据槽完整度 >= 65% 或已追问超过 5 轮
     → action=function_call，调用 law_lookup 检索法条，然后进入分析阶段

2. **分析阶段（ANALYZING / TOOL_USE）**：
   - 收到 law_lookup 结果后 → action=function_call 继续检索（如果结果不足）或进入生成
   - 法条检索结果充足 → 进入报告生成

3. **报告生成（GENERATING）**：
   - 基于证据槽 + 法条检索结果，生成完整的 8 模块分析报告
   - 报告必须包含：案情摘要、赔偿估算、法律依据、证据清单、行动路径、风险提示、法律定性、免责声明

4. **反思校验（REVIEWING）**：
   - 检查报告是否包含法条引用、证据建议、免责声明
   - 通过 → action=final_answer
   - 不通过 → 重新生成（最多 1 次）

5. **特殊转出（REFERRING_OUT）**：
   - 如果用户提到工伤认定、社保补缴、伤残等级鉴定 → 引导用户先走行政前置程序

6. **终态处理（FINALIZING）**：
   - 用户追问新问题 → 回到信息收集阶段
   - 用户要求生成文书 → 调用文档生成工具

## 对话历史
{conversation_history}
"""


# ============================================================
# 证据槽摘要构建
# ============================================================
def _build_slots_summary(sess: AgentSession) -> str:
    """把证据槽转成 LLM 可读的摘要"""
    done, total, ratio = sess.evidence_slots.completion_ratio()
    questions = sess.evidence_slots.next_questions_to_ask(3)

    lines = [f"信息完整度：{done}/{total}（{ratio:.0%}）"]

    flat = sess.evidence_slots.to_flat_dict(True)
    if flat:
        lines.append("已获取的信息：")
        for slot in sess.evidence_slots.list_slots():
            if slot.state.value in ("provided", "derived") and slot.value is not None:
                tag = "（推断，待确认）" if slot.state.value == "derived" else ""
                lines.append(f"  - {slot.label}: {slot.value}{tag}")

    missing_slots = sess.evidence_slots.list_slots(state=None)
    missing = [s for s in missing_slots if s.state.value == "missing"]
    if missing:
        lines.append("缺失的关键信息：")
        for s in missing[:5]:
            lines.append(f"  - {s.label}（优先级: {s.priority.name}）")

    if questions:
        lines.append("建议追问的问题：")
        for i, q in enumerate(questions, 1):
            tag = "（推断，待确认）" if q.state.value == "derived" else ""
            lines.append(f"  {i}. {q.label}{tag}")

    return "\n".join(lines)


# ============================================================
# 状态描述构建
# ============================================================
def _build_state_description(sess: AgentSession) -> str:
    """把当前 FSM 状态 + 跃迁历史转成 LLM 可读的描述"""
    state = sess.fsm.current_state
    history = sess.fsm.transitions

    state_descriptions = {
        "init": "刚创建会话，等待用户描述案情",
        "understanding": "正在理解用户描述，抽取结构化案情信息",
        "collecting": "信息不足，需要向用户追问关键信息",
        "analyzing": "信息充足，正在检索相关法律依据",
        "tool_use": "正在调用工具（法条检索 / 赔偿计算 / 文书生成）",
        "generating": "正在生成分析报告",
        "reviewing": "正在反思校验报告质量",
        "finalizing": "已给出最终结论，等待用户后续操作",
        "referring_out": "已引导用户走行政前置程序",
    }

    lines = [
        f"当前阶段：{state.value} — {state_descriptions.get(state.value, '未知')}",
    ]

    if history:
        lines.append("已完成的状态跃迁：")
        for ev in history[-5:]:  # 最近 5 次跃迁
            lines.append(f"  {ev.from_state.value} → {ev.to_state.value}（{ev.reason[:50]}）")

    return "\n".join(lines)


# ============================================================
# 对话历史格式化
# ============================================================
def _build_conversation_history(sess: AgentSession) -> str:
    """把短期记忆格式化成文本"""
    items = sess.short_term.read()
    if not items:
        return "（暂无对话历史）"

    dialogue = [it for it in items if it.role in ("user", "assistant")]
    if not dialogue:
        return "（暂无对话历史）"

    lines = []
    for it in dialogue[-12:]:  # 最近 6 轮对话
        role = "用户" if it.role == "user" else "Agent"
        content = str(it.content)
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"[{role}] {content}")

    return "\n".join(lines)


# ============================================================
# 公开 API：构建完整的 Agent 上下文
# ============================================================
def build_agent_context(sess: AgentSession) -> list[dict]:
    """
    构建发给 LLM 的完整 messages 列表。

    返回格式：
    [
      {"role": "system", "content": "完整的 System Prompt（含状态+证据+工具+规则）"},
      {"role": "user", "content": "用户最新一条消息"}
    ]

    注意：对话历史已经嵌入 System Prompt 中，所以这里只返回 system + 最新 user 消息。
    """
    state_desc = _build_state_description(sess)
    slots_summary = _build_slots_summary(sess)
    tools_desc = sess.tools.summarize_for_llm_text() if hasattr(sess, 'tools') else "（无可用工具）"
    history = _build_conversation_history(sess)

    system_content = _SYSTEM_PROMPT_TEMPLATE.format(
        current_state_description=state_desc,
        evidence_slots_summary=slots_summary,
        tools_description=tools_desc,
        conversation_history=history,
    ) + _DECISION_FORMAT

    # 最新一条用户消息从 ShortTermMemory 取
    latest_user = sess.short_term.peek_latest("user")
    user_content = str(latest_user.content) if latest_user else ""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ============================================================
# 辅助：解析 LLM 返回的 JSON 决策
# ============================================================
def parse_agent_decision(llm_text: str) -> dict | None:
    """
    从 LLM 回复中提取 JSON 决策块。

    返回 None 表示解析失败（调用方应做兜底处理）。
    """
    import json
    import re

    if not llm_text:
        return None

    # 尝试匹配 ```json ... ``` 代码块
    fence = re.search(r"```json\s*(.*?)```", llm_text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        # 没有代码块，尝试找第一个 { ... } 对
        start = llm_text.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        end = start
        for i in range(start, len(llm_text)):
            ch = llm_text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = llm_text[start:end]

    try:
        decision = json.loads(candidate)
        if "action" not in decision:
            return None
        return decision
    except (json.JSONDecodeError, TypeError):
        return None