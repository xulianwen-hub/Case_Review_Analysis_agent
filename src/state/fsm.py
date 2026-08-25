"""
src.state.fsm —— 对话有限状态机（FSM）

💡 面试讲解点（灵魂拷问：为什么选 FSM？不选 ReAct/Graph/纯 LLM 路由？）
    · 先把三个主流方案讲清楚，再说我们的选择：

    方案 A：纯 ReAct（LLM 每次自己决定 next action）
        ✘ 劳动纠纷场景流程非常刚性（必须先问清楚工资工龄，才能谈赔偿），
            ReAct 很容易"跳步骤" —— 比如用户说「老板把我开了」，LLM 可能直接给结论跳过证据收集，
            导致最终赔偿金额错得离谱。可解释性差，失败了不知道是哪一步出的错。

    方案 B：StateGraph（LangGraph 式的有向图）
        ✔ 非常灵活，支持节点任意跳转、条件边、并行节点
        ✘ 学习曲线陡 + 对我们的场景"杀鸡用牛刀"，
            劳动纠纷咨询 90% 是线性的「收集信息 → 分析 → 计算 → 出报告」，
            真正需要分支跳转的地方非常有限。

    方案 C：FSM 有限状态机（我们这个实现）✔✔✔
        ✔ 状态集合有限 + 跃迁边显式定义 = 100% 可预测，不会乱跳
        ✔ on_enter / on_exit 钩子 + 每个状态的 process() 方法，代码极度可读
        ✔ 非法跃迁自动拦截（比如 FINAL 再收到消息不能回 ANALYZING）
        ✔ 调试：出错了打印「用户发了 X → 从 S1 跃迁到 S2 因为条件 C 满足」，
            一个人就能把复杂 Agent 流程 debug 出来
        → 对"咨询类 Agent（有明确前置信息需求）"来说，FSM 是性价比最高的方案。

    ➕ 我们的设计留了扩展位：transition 函数目前是规则触发，
      未来完全可以把「是否应该跃迁」这一步交给 LLM 判断 → 相当于 Hybrid FSM，
      既保留 FSM 的可预测性，又有 LLM 的灵活性。
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ============================================================
# 枚举：9 个对话状态（劳动纠纷咨询的标准业务流程）
# ============================================================
class AgentState(Enum):
    INIT = "init"                       # 刚创建，还没收到第一条用户消息
    UNDERSTANDING = "understanding"     # 用 LLM 结构化抽取案情 → 写证据槽
    COLLECTING_EVIDENCE = "collecting"  # 证据槽 MISSING 项太多 → 逐个追问用户
    ANALYZING = "analyzing"             # 信息够了 → 调 RAG 法条/案例检索
    TOOL_USE = "tool_use"               # 调工具（赔偿金计算器/文档生成 等）
    GENERATING = "generating"           # 调 LLM 生成最终 8 模块报告
    REVIEWING = "reviewing"             # 反思/一致性校验（数字 vs 法条 vs 证据）
    FINALIZING = "finalizing"           # 输出最终结论 + 可选生成附件
    REFERRING_OUT = "referring_out"     # 不适合 Agent 处理的场景（工伤认定/社保补缴）
    # 扩展位：将来加 CHATTING（闲聊） / CALCULATING_DETAIL（用户追问某项具体计算）

    @property
    def is_terminal(self) -> bool:
        """终态：这两个状态之后应该直接输出或引导用户开始新会话"""
        return self in (AgentState.FINALIZING, AgentState.REFERRING_OUT)


# ============================================================
# 数据结构：一次跃迁的上下文记录（可追溯性 / 调试 / 用户可解释）
# ============================================================
@dataclass
class TransitionEvent:
    from_state: AgentState
    to_state: AgentState
    reason: str                     # 人类可读的跃迁原因（debug + 将来给用户解释 Agent 行为）
    output: Any = None              # 状态机本轮给用户的回复文本/结构化片段
    trigger: str = "user_message"   # 触发源：user_message / timeout / tool_result / orchestrator
    metadata: dict = field(default_factory=dict)


# ============================================================
# 核心：有限状态机
# ============================================================
class DialogueFSM:
    """
    标准 Moore 型 FSM：
        每个状态 S 有：
            on_enter(ctx)     进入状态时做的准备（可选覆盖）
            process(ctx, user_input) → next_state, response  本状态的核心处理
            on_exit(ctx)      离开状态时的清理（可选覆盖）
        跃迁有：
            合法性表 ALLOWED_TRANSITIONS — 非法跃迁直接抛 AssertionError
            transition() 统一入口，返回 TransitionEvent（可追溯）
    """

    # ---------- 合法跃迁表：从哪些状态可以跳到哪些状态 ----------
    # 面试讲解：这张表就是 FSM 的「安全网」，防止 LLM 编排时的乱跳
    ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
        # 用户首次发消息
        AgentState.INIT: {
            AgentState.UNDERSTANDING,
        },
        # 理解完成 → 要么缺信息去追问，要么够信息直接分析
        AgentState.UNDERSTANDING: {
            AgentState.COLLECTING_EVIDENCE,
            AgentState.ANALYZING,
            AgentState.REFERRING_OUT,
        },
        # 追问一轮 → 再回到理解（循环），或者信息齐了去分析
        AgentState.COLLECTING_EVIDENCE: {
            AgentState.UNDERSTANDING,
            AgentState.ANALYZING,
        },
        # 分析完成 → 要么需要用工具，要么直接生成报告
        AgentState.ANALYZING: {
            AgentState.TOOL_USE,
            AgentState.GENERATING,
        },
        # 工具完成 → 生成报告（或再分析，如果工具返回说缺字段）
        AgentState.TOOL_USE: {
            AgentState.GENERATING,
            AgentState.ANALYZING,       # 工具反馈缺字段 → 重新分析
            AgentState.COLLECTING_EVIDENCE,  # 工具明确要求用户补充信息
        },
        # 生成完成 → 反思校验
        AgentState.GENERATING: {
            AgentState.REVIEWING,
        },
        # 反思通过 → 终态；反思失败 → 回退生成（最多重试2次由 orchestrator 控制）
        AgentState.REVIEWING: {
            AgentState.FINALIZING,
            AgentState.GENERATING,
        },
        # 终态也允许回来（用户接着追问新问题），但设计上 FINALIZING 之后 Orchestrator 会给用户
        # 「是否继续追问/需要我生成文书吗？」的选项
        AgentState.FINALIZING: {
            AgentState.UNDERSTANDING,   # 新的一轮问题（同一会话）
            AgentState.TOOL_USE,        # 用户说「帮我生成仲裁申请书吧」
        },
        # 转出指引也允许回来（用户说「哦我有工伤认定书了」）
        AgentState.REFERRING_OUT: {
            AgentState.UNDERSTANDING,
        },
    }
    # 面试讲解：「全局安全边」—— 无论当前处于什么状态，只要命中工伤/社保补缴
    # 关键词，都允许立刻跳去 REFERRING_OUT（转行政前置程序）。
    # 这类「与流程无关的高优先级转出」用全局边一次性声明，避免每写一个新状态
    # 都要记得加 REFERRING_OUT。
    for _s in AgentState:
        ALLOWED_TRANSITIONS.setdefault(_s, set()).add(AgentState.REFERRING_OUT)

    def __init__(self, initial_state: AgentState = AgentState.INIT):
        self._current_state = initial_state
        self._history: list[TransitionEvent] = []
        # 钩子字典（允许 Orchestrator 动态挂行为，不需要子类化 FSM）
        self._on_enter: dict[AgentState, list[Callable]] = {}
        self._on_exit: dict[AgentState, list[Callable]] = {}

    # ============================================================
    # 当前状态：只读属性 + 跃迁历史
    # ============================================================
    @property
    def current_state(self) -> AgentState:
        return self._current_state

    @property
    def transitions(self) -> list[TransitionEvent]:
        return list(self._history)

    def can_transition_to(self, target: AgentState) -> bool:
        allowed = self.ALLOWED_TRANSITIONS.get(self._current_state, set())
        return target in allowed

    # ============================================================
    # 钩子注册（可选使用：Orchestrator 可以在这里挂 "进入ANALYZING就调RAG"）
    # ============================================================
    def on_enter(self, state: AgentState, callback: Callable) -> None:
        self._on_enter.setdefault(state, []).append(callback)

    def on_exit(self, state: AgentState, callback: Callable) -> None:
        self._on_exit.setdefault(state, []).append(callback)

    # ============================================================
    # 核心：跃迁
    # ============================================================
    def transition(self,
                   target: AgentState,
                   *,
                   reason: str,
                   output: Any = None,
                   trigger: str = "user_message",
                   ctx: Optional[dict] = None) -> TransitionEvent:
        """
        统一跃迁入口：
            1) 校验合法性（ALLOWED_TRANSITIONS 不允许就抛 AssertionError）
            2) 执行 from_state 的 on_exit 钩子
            3) 切换 state
            4) 执行 to_state 的 on_enter 钩子
            5) 记录 TransitionEvent 到 history 并返回
        """
        from_state = self._current_state
        if not self.can_transition_to(target):
            raise AssertionError(
                f"FSM 非法跃迁: {from_state.value} → {target.value} 不在允许表中。"
                f" 从 {from_state.value} 允许跳往: "
                f"{[s.value for s in self.ALLOWED_TRANSITIONS.get(from_state, set())]}"
            )
        # --- on_exit ---
        for cb in self._on_exit.get(from_state, []):
            try:
                cb(state=from_state, ctx=ctx or {})
            except Exception:
                pass  # 钩子失败不影响核心跃迁（钩子默认是观测类/副作用）
        # --- 切换 ---
        self._current_state = target
        # --- on_enter ---
        for cb in self._on_enter.get(target, []):
            try:
                cb(state=target, ctx=ctx or {})
            except Exception:
                pass
        event = TransitionEvent(
            from_state=from_state, to_state=target,
            reason=reason, output=output, trigger=trigger,
        )
        self._history.append(event)
        return event

    def reset(self) -> None:
        self._current_state = AgentState.INIT
        self._history.clear()

    # ============================================================
    # 调试：漂亮打印跃迁链
    # ============================================================
    def pretty_transitions(self) -> str:
        if not self._history:
            return "（FSM 还没跃迁过）"
        lines = ["FSM 跃迁轨迹："]
        for i, ev in enumerate(self._history, 1):
            lines.append(
                f"  [{i}] {ev.from_state.value:>11} ──▶ {ev.to_state.value:<13}"
                f"   原因: {ev.reason[:60]}{'…' if len(ev.reason)>60 else ''}"
            )
        lines.append(f"  📍 当前状态: {self._current_state.value}")
        return "\n".join(lines)