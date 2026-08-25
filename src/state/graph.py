"""
src.state.graph —— LangGraph Agent Graph（企业级状态管理 + Checkpoint）

用 LangGraph 的 StateGraph + Checkpointer 替换自研 FSM + Agent Loop。

核心优势：
    1. 内置 Checkpoint：每次节点执行后自动保存状态快照，进程重启可恢复
    2. 图结构天然支持复杂分支和循环（比线性 FSM 更灵活）
    3. 状态 Schema 由 TypedDict 定义，类型安全 + 自动序列化
    4. 支持 Human-in-the-Loop（interrupt/resume）
    5. 天然支持流式输出（stream）

图结构：
    START → extract_evidence → decide →
        ├─ ask_user → END（等待用户下一轮输入）
        ├─ search_law → decide（循环：检索完再决策）
        ├─ generate_report → review →
        │     ├─ finalize → END
        │     └─ generate_report（反思失败，重试）
        └─ finalize → END（已有报告，直接终态）

注意：
    · 对话历史由 LangGraph 的 messages 字段管理（add_messages reducer 自动追加）
    · 证据槽以 dict 形式存储在 state 中，使用时重建 EvidenceSlotMemory
    · 每次 graph.invoke() 调用，checkpointer 自动在节点间保存快照
"""
from __future__ import annotations

import json
import logging
from typing import TypedDict, Annotated, Literal, Optional, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

logger = logging.getLogger(__name__)


# ============================================================
# State Schema —— 所有字段必须可 JSON 序列化
# ============================================================
class AgentState(TypedDict, total=False):
    """
    LangGraph Agent 状态定义。

    messages:      对话历史（add_messages reducer 自动追加，不覆盖）
    session_id:    会话 ID
    evidence_slots: 证据槽（dict 形式，按 slot_name 索引）
    current_phase:  当前阶段（字符串，替代 FSM 枚举）
    next_action:    LLM 决策的下一步动作（ask_user|search_law|generate_report|finalize）
    collect_rounds: 追问轮数计数
    tool_results:   工具调用结果缓存（list of dict）
    response_text:  返回给用户的文本
    suggested_actions: 建议用户的后续操作
    user_meta:      用户元信息
    review_retry_count: 反思重试计数
    """
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    evidence_slots: dict[str, dict]
    current_phase: str
    next_action: str
    collect_rounds: int
    tool_results: list[dict]
    response_text: str
    suggested_actions: list[str]
    user_meta: dict
    review_retry_count: int


# ============================================================
# 默认初始状态
# ============================================================
def get_initial_state(session_id: str = "", user_meta: Optional[dict] = None) -> dict:
    return {
        "messages": [],
        "session_id": session_id,
        "evidence_slots": {},
        "current_phase": "init",
        "next_action": "extract_evidence",
        "collect_rounds": 0,
        "tool_results": [],
        "response_text": "",
        "suggested_actions": [],
        "user_meta": user_meta or {},
        "review_retry_count": 0,
    }


# ============================================================
# 辅助：从 state dict 重建 EvidenceSlotMemory
# ============================================================
def _rebuild_evidence_slots(state: AgentState):
    """从 state 中的 dict 重建 EvidenceSlotMemory 对象"""
    from ..memory.evidence_slots import EvidenceSlotMemory, SlotState, SlotPriority

    slots = EvidenceSlotMemory()
    for name, data in state.get("evidence_slots", {}).items():
        try:
            priority = SlotPriority(data.get("priority", "P1"))
        except (ValueError, TypeError):
            priority = SlotPriority.P1
        try:
            slot_state = SlotState(data.get("state", "missing"))
        except (ValueError, TypeError):
            slot_state = SlotState.MISSING
        slots.update(
            name,
            value=data.get("value"),
            label=data.get("label", name),
            state=slot_state,
            priority=priority,
            ask_hint=data.get("ask_hint", ""),
        )
    return slots


def _serialize_evidence_slots(slots) -> dict[str, dict]:
    """把 EvidenceSlotMemory 序列化为 dict"""
    result = {}
    for slot in slots.list_slots():
        result[slot.name] = {
            "label": slot.label,
            "value": slot.value,
            "state": slot.state.value,
            "priority": slot.priority.name,
            "ask_hint": slot.ask_hint,
        }
    return result


# ============================================================
# 辅助：LLM 决策解析
# ============================================================
def _parse_llm_decision(llm_text: str) -> dict:
    """从 LLM 回复中提取 JSON 决策"""
    import re
    if not llm_text:
        return {"action": "ask_user", "reasoning": "LLM 返回为空"}

    fence = re.search(r"```json\s*(.*?)```", llm_text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        start = llm_text.find("{")
        if start < 0:
            return {"action": "ask_user", "reasoning": "未找到 JSON 决策"}
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
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return {"action": "ask_user", "reasoning": "JSON 解析失败"}


# ============================================================
# 决策 System Prompt
# ============================================================
_DECIDE_PROMPT = """你是一个劳动法咨询 Agent 的决策模块。

当前阶段: {current_phase}
证据完整度: {completion_ratio:.0%}（{done}/{total}）
追问轮数: {collect_rounds}
可用工具: law_lookup, evidence_extract, report_generate, report_review

## 决策规则
1. 如果当前阶段是 init/understanding/collecting，且证据完整度 < 65% 且追问轮数 < 5:
   → action=ask_user, 追问缺失的关键信息
2. 如果证据完整度 >= 65% 或追问轮数 >= 5:
   → action=search_law, 检索法条
3. 如果已经检索了法条，但还未生成报告:
   → action=generate_report
4. 如果报告已生成并通过校验:
   → action=finalize
5. 如果报告已生成但未校验:
   → action=review

## 输出格式
```json
{{"action": "ask_user|search_law|generate_report|review|finalize", "reasoning": "一句话原因"}}
```

## 证据槽状态
{evidence_summary}
"""


# ============================================================
# LangGraph 节点函数
# ============================================================
class LaborLawGraph:
    """
    劳动法咨询 Agent 的 LangGraph 图。

    使用方式：
        graph = LaborLawGraph(extractor, report_gen, reviewer, tools, llm, db_path="checkpoints.db")
        result = graph.run(session_id="session_123", user_input="我被辞退了")
        # result["response_text"] → 给用户的回复
        # 状态自动 checkpoint 到 SQLite

    面试亮点：
        · 用 LangGraph 的 StateGraph 替代自研 FSM，获得内置 Checkpoint + 流式 + 可观测性
        · 业务逻辑（extractor/report_gen/reviewer）完全复用，只替换了状态管理层
        · 图结构天然支持复杂分支和循环，比线性 FSM 更灵活
    """

    def __init__(self,
                 extractor,
                 report_generator,
                 reviewer,
                 tools,
                 llm,
                 db_path: Optional[str] = None,
                 max_collect_rounds: int = 5,
                 completion_threshold: float = 0.65):
        self.extractor = extractor
        self.report_generator = report_generator
        self.reviewer = reviewer
        self.tools = tools
        self.llm = llm
        self.max_collect_rounds = max_collect_rounds
        self.completion_threshold = completion_threshold

        # Checkpointer: SQLite 优先，回退到内存
        if db_path:
            self.checkpointer = SqliteSaver.from_conn_string(db_path)
        else:
            self.checkpointer = MemorySaver()

        self.graph = self._build()

    # ============== 图构建 ==============
    def _build(self):
        builder = StateGraph(AgentState)

        # 注册节点
        builder.add_node("extract_evidence", self._node_extract_evidence)
        builder.add_node("decide", self._node_decide)
        builder.add_node("ask_user", self._node_ask_user)
        builder.add_node("search_law", self._node_search_law)
        builder.add_node("generate_report", self._node_generate_report)
        builder.add_node("review", self._node_review)
        builder.add_node("finalize", self._node_finalize)

        # 定义边
        builder.add_edge(START, "extract_evidence")
        builder.add_edge("extract_evidence", "decide")
        builder.add_conditional_edges(
            "decide", self._route_after_decide,
            {"ask_user": "ask_user", "search_law": "search_law",
             "generate_report": "generate_report", "review": "review",
             "finalize": "finalize"},
        )
        builder.add_edge("search_law", "decide")  # 检索完回到决策节点
        builder.add_edge("ask_user", END)
        builder.add_edge("generate_report", "review")
        builder.add_conditional_edges(
            "review", self._route_after_review,
            {"finalize": "finalize", "generate_report": "generate_report"},
        )
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=self.checkpointer)

    # ============== 路由函数 ==============
    def _route_after_decide(self, state: AgentState) -> str:
        action = state.get("next_action", "ask_user")
        valid = {"ask_user", "search_law", "generate_report", "review", "finalize"}
        return action if action in valid else "ask_user"

    def _route_after_review(self, state: AgentState) -> str:
        retry = state.get("review_retry_count", 0)
        if retry >= 2:
            return "finalize"
        return "finalize" if state.get("next_action") == "finalize" else "generate_report"

    # ============== 节点：extract_evidence ==============
    def _node_extract_evidence(self, state: AgentState) -> dict:
        """从用户最新消息中抽取结构化证据"""
        messages = state.get("messages", [])
        if not messages:
            return {"current_phase": "understanding"}

        # 取最后一条用户消息
        user_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_text = m.content
                break

        if not user_text:
            return {"current_phase": "understanding"}

        # 重建 EvidenceSlotMemory 并抽取
        slots = _rebuild_evidence_slots(state)
        extracted = self.extractor.extract(user_text, slots)
        if extracted:
            slots.batch_update(extracted, source="langgraph_extract")

        return {
            "evidence_slots": _serialize_evidence_slots(slots),
            "current_phase": "understanding",
        }

    # ============== 节点：decide ==============
    def _node_decide(self, state: AgentState) -> dict:
        """LLM 决定下一步动作"""
        slots = _rebuild_evidence_slots(state)
        done, total, ratio = slots.completion_ratio()
        collect_rounds = state.get("collect_rounds", 0)
        current_phase = state.get("current_phase", "init")
        tool_results = state.get("tool_results", [])
        has_tool_result = len(tool_results) > 0
        review_retry = state.get("review_retry_count", 0)

        # 规则兜底（快速路径，不调 LLM）
        if current_phase in ("init", "understanding", "collecting"):
            if ratio >= self.completion_threshold or collect_rounds >= self.max_collect_rounds:
                if not has_tool_result:
                    return {"next_action": "search_law", "current_phase": "analyzing"}
                else:
                    return {"next_action": "generate_report", "current_phase": "generating"}
            else:
                return {"next_action": "ask_user", "current_phase": "collecting",
                        "collect_rounds": collect_rounds + 1}

        if current_phase == "analyzing":
            if has_tool_result:
                return {"next_action": "generate_report", "current_phase": "generating"}
            return {"next_action": "search_law"}

        if current_phase == "generating":
            if review_retry >= 2:
                return {"next_action": "finalize", "current_phase": "finalizing"}
            return {"next_action": "review", "current_phase": "reviewing"}

        if current_phase == "reviewing":
            return {"next_action": "finalize", "current_phase": "finalizing"}

        # 默认：信息不够就追问
        if ratio < self.completion_threshold and collect_rounds < self.max_collect_rounds:
            return {"next_action": "ask_user", "current_phase": "collecting",
                    "collect_rounds": collect_rounds + 1}

        return {"next_action": "search_law", "current_phase": "analyzing"}

    # ============== 节点：ask_user ==============
    def _node_ask_user(self, state: AgentState) -> dict:
        """构建追问用户的响应"""
        slots = _rebuild_evidence_slots(state)
        done, total, ratio = slots.completion_ratio()
        questions = slots.next_questions_to_ask(3)

        lines = [f"✅ 已收到你的描述（当前信息完整度：{done}/{total} 关键项，{ratio:.0%}）。"]
        if questions:
            lines.append(f"\n❓ 为了给出准确的赔偿金额和方案，我还需要确认 {len(questions)} 件事：")
            for i, q in enumerate(questions, 1):
                tag = "（推断，待确认）" if q.state.value == "derived" else "（缺失）"
                lines.append(f"   {i}. {q.label}{tag}")
                if q.ask_hint:
                    lines.append(f"      → {q.ask_hint}")

        response = "\n".join(lines)
        return {
            "response_text": response,
            "suggested_actions": ["直接回答上面的问题", "说「先给我一个大致结论吧」跳过追问"],
            "messages": [AIMessage(content=response)],
        }

    # ============== 节点：search_law ==============
    def _node_search_law(self, state: AgentState) -> dict:
        """调用 RAG 工具检索法条"""
        messages = state.get("messages", [])
        user_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_text = m.content
                break

        # 构建检索 query
        slots = _rebuild_evidence_slots(state)
        flat = slots.to_flat_dict(True)
        query_parts = [user_text]
        for key in ("termination_reason", "dispute_type"):
            if flat.get(key):
                query_parts.append(str(flat[key]))
        query = " ".join(query_parts)

        tool_result = self.tools.run("law_lookup", query=query)
        result_dict = {
            "tool_name": "law_lookup",
            "success": tool_result.success,
            "output": tool_result.output if tool_result.success else None,
            "error": tool_result.error if not tool_result.success else "",
        }

        return {
            "tool_results": [result_dict],
            "current_phase": "analyzing",
            "messages": [ToolMessage(
                content=f"法条检索完成: success={tool_result.success}",
                tool_call_id="law_lookup",
            )],
        }

    # ============== 节点：generate_report ==============
    def _node_generate_report(self, state: AgentState) -> dict:
        """生成 8 模块分析报告"""
        messages = state.get("messages", [])
        user_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_text = m.content
                break

        slots = _rebuild_evidence_slots(state)
        tool_results = state.get("tool_results", [])
        tool_output = None
        if tool_results:
            last = tool_results[-1]
            if last.get("success"):
                tool_output = last.get("output")

        # 用已有的 report_generator 生成
        report = self.report_generator.generate(user_text, slots, tool_output)

        return {
            "response_text": report,
            "current_phase": "generating",
            "messages": [AIMessage(content=report)],
            "review_retry_count": state.get("review_retry_count", 0),
        }

    # ============== 节点：review ==============
    def _node_review(self, state: AgentState) -> dict:
        """反思校验报告质量"""
        report = state.get("response_text", "")
        slots = _rebuild_evidence_slots(state)
        tool_results = state.get("tool_results", [])
        tool_output = None
        if tool_results:
            last = tool_results[-1]
            if last.get("success"):
                tool_output = last.get("output")

        review = self.reviewer.review(report, slots, tool_output)
        retry_count = state.get("review_retry_count", 0)

        if review.passed or retry_count >= 1:
            return {
                "next_action": "finalize",
                "current_phase": "finalizing",
                "messages": [SystemMessage(content=f"反思校验通过: {review.reason}")],
            }

        # 未通过，重试
        hint = "；".join(review.suggestions) if review.suggestions else "请在报告中明确引用法条编号"
        messages = state.get("messages", [])
        user_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_text = m.content
                break

        enhanced_input = f"{user_text}；[反思提示：{hint}]"
        slots = _rebuild_evidence_slots(state)
        report = self.report_generator.generate(enhanced_input, slots, tool_output)

        return {
            "response_text": report,
            "next_action": "generate_report",
            "current_phase": "generating",
            "review_retry_count": retry_count + 1,
            "messages": [
                SystemMessage(content=f"反思校验未通过: {review.reason}，重试第{retry_count + 1}次"),
                AIMessage(content=report),
            ],
        }

    # ============== 节点：finalize ==============
    def _node_finalize(self, state: AgentState) -> dict:
        """终态：组装最终响应"""
        response = state.get("response_text", "分析完成。")
        disclaimer = "\n\n---\n⚠️ 以上分析仅供参考，不构成法律意见。如有复杂情况，请咨询专业律师。"
        if disclaimer not in response:
            response += disclaimer

        return {
            "response_text": response,
            "current_phase": "finalizing",
            "suggested_actions": [
                "生成仲裁申请书 Word",
                "补充更多细节重新计算",
                "开始一个新的咨询（说「重新开始」）",
            ],
            "messages": [AIMessage(content=response)],
        }

    # ============== 公开 API：运行图 ==============
    def run(self, session_id: str, user_input: str,
            user_meta: Optional[dict] = None) -> dict:
        """
        运行 Agent 图。

        Args:
            session_id: 会话 ID（也是 LangGraph 的 thread_id）
            user_input: 用户输入
            user_meta: 用户元信息

        Returns:
            dict with keys: response_text, session_id, current_phase, suggested_actions
        """
        config = {"configurable": {"thread_id": session_id}}

        # 尝试从 checkpoint 恢复状态
        try:
            current_state = self.graph.get_state(config)
            has_checkpoint = current_state is not None and current_state.values
        except Exception:
            has_checkpoint = False

        if has_checkpoint:
            # 从 checkpoint 恢复，追加新消息
            updated = {"messages": [HumanMessage(content=user_input)]}
        else:
            # 首次对话，创建初始状态
            initial = get_initial_state(session_id, user_meta)
            initial["messages"] = [HumanMessage(content=user_input)]
            updated = initial

        logger.info(
            f"[LangGraph] run session={session_id}, "
            f"has_checkpoint={has_checkpoint}, "
            f"user_input={user_input[:50]}..."
        )

        # 执行图
        final_state = self.graph.invoke(updated, config)

        return {
            "response_text": final_state.get("response_text", ""),
            "session_id": session_id,
            "current_phase": final_state.get("current_phase", "init"),
            "suggested_actions": final_state.get("suggested_actions", []),
            "evidence_slots": final_state.get("evidence_slots", {}),
            "collect_rounds": final_state.get("collect_rounds", 0),
        }

    # ============== 辅助：获取当前状态快照（调试/监控用） ==============
    def get_state_snapshot(self, session_id: str) -> Optional[dict]:
        """获取某个会话的当前状态快照"""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.graph.get_state(config)
            if state and state.values:
                vals = dict(state.values)
                vals["next_nodes"] = state.next if state.next else []
                return vals
        except Exception as e:
            logger.warning(f"[LangGraph] get_state_snapshot failed: {e}")
        return None

    def get_state_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """获取某个会话的状态历史（所有 checkpoint）"""
        config = {"configurable": {"thread_id": session_id}}
        history = []
        try:
            for snapshot in self.graph.get_state_history(config, limit=limit):
                history.append({
                    "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id", ""),
                    "phase": snapshot.values.get("current_phase", "") if snapshot.values else "",
                    "next": snapshot.next if snapshot.next else [],
                })
        except Exception as e:
            logger.warning(f"[LangGraph] get_state_history failed: {e}")
        return history


# ============================================================
# 工厂函数：创建默认图（SQLite 持久化）
# ============================================================
def create_default_graph(extractor, report_generator, reviewer, tools, llm,
                         db_path: str = "data/checkpoints.db") -> LaborLawGraph:
    """
    创建默认的 LaborLawGraph 实例（使用 SQLite checkpoint）。

    如果 SQLite 不可用，自动降级为内存 checkpoint。
    """
    import os
    if db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        return LaborLawGraph(extractor, report_generator, reviewer, tools, llm, db_path=db_path)
    except Exception as e:
        logger.warning(f"SQLite checkpoint 初始化失败 ({e})，降级为内存 checkpoint")
        return LaborLawGraph(extractor, report_generator, reviewer, tools, llm, db_path=None)