"""
src.memory.evidence_slots —— 证据槽记忆（劳动纠纷 Agent 的核心特色结构化记忆）

💡 面试讲解点（亮点！体现「领域 Agent 和通用 Chatbot 的本质区别」）
    · 是什么：把劳动纠纷咨询所需的 20+ 项关键信息，抽象成一个个「槽位」，
             每个槽有三态 + 置信度 + 追问优先级
    · 为什么这样设计：
        通用对话机器人 → 给用户东拉西扯
        专业 Agent → 知道「这个案子必须知道这 8 件事才能给准确结论」
          → 证据槽驱动的追问：缺什么问什么，不多问、不重复问
    · 和普通对话记忆的区别：
        对话记忆是时序的（用户说的话按时间堆着）
        证据槽是**结构化**的（不管用户在第几轮提到工资，都写入同一个 salary 槽）
        → 是领域知识的沉淀，对话只是填充槽的手段
    · 追问优先级算法：
        「计算赔偿必须的」>「可选增强证据」>「锦上添花」
        置信度越低越先问；同级按「单次提问能问最多槽位」排序
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any

from .base import BaseMemory, MemoryItem


# ============================================================
# 枚举：槽位状态
# ============================================================
class SlotState(Enum):
    MISSING = "missing"       # 完全缺失（必须追问）
    DERIVED = "derived"       # 从对话中推断出来，未被用户显式确认（建议追问确认）
    PROVIDED = "provided"     # 用户显式提供，可使用
    INAPPLICABLE = "n/a"      # 对本案情不适用（比如用户没提加班费，加班费槽 = n/a，不问）


# ============================================================
# 枚举：追问优先级（数字越小越先问）
# ============================================================
class SlotPriority(Enum):
    P0_CRITICAL = 0     # 赔偿计算必需项（缺了算不了数字）
    P1_IMPORTANT = 1    # 结论强依赖项（违法解除 vs 合法解除）
    P2_SUPPORT = 2      # 证据链补强（胜率提升项）
    P3_OPTIONAL = 3     # 锦上添花


# ============================================================
# 数据结构：槽位定义
# ============================================================
@dataclass
class EvidenceSlot:
    key: str                               # 唯一 key，如 "monthly_salary"
    label: str                             # 给用户看的中文名，如 "月平均应发工资"
    state: SlotState = SlotState.MISSING
    value: Any = None                      # 存储的值（字符串/数字/日期）
    confidence: float = 0.0                # 0.0 - 1.0，DERIVED 时通常 0.5-0.8，PROVIDED 时 1.0
    priority: SlotPriority = SlotPriority.P1_IMPORTANT
    ask_hint: str = ""                     # 追问时用的话术模板
    category: str = "general"              # 分类：id / work / salary / evidence / termination / social_insurance
    source: str = ""                       # 值的来源：user_message / llm_derive / tool_result


@dataclass
class SlotUpdateRecord:
    """槽位被更新的一次记录，用于「解释 Agent 为什么知道这个信息」（可追溯性）"""
    key: str
    old_state: SlotState
    new_state: SlotState
    old_value: Any
    new_value: Any
    reason: str                              # 为什么这次更新（"用户在第3轮说工资1万2"等）


# ============================================================
# 核心：证据槽管理器
# ============================================================
class EvidenceSlotMemory(BaseMemory):
    """
    证据槽记忆（领域 Agent 核心）：
        · 初始化时注册一份「劳动纠纷标准槽位清单」
        · 每轮对话后用 LLM 抽取结果调用 update_slot()
        · 调用 next_questions_to_ask() 得到本轮应追问的 1-3 个问题
    """

    # ============================================================
    # 标准槽位定义（V1：覆盖 80% 常见情形）
    # ============================================================
    STANDARD_SLOTS = [
        # --- 身份 & 劳动关系 P0 ---
        EvidenceSlot("company_city", "公司所在城市",
                     priority=SlotPriority.P0_CRITICAL, category="id",
                     ask_hint="公司是在哪个城市注册/办公的？（影响最低工资和地区裁审口径）"),
        EvidenceSlot("employment_start", "入职日期（精确到月）",
                     priority=SlotPriority.P0_CRITICAL, category="work",
                     ask_hint="你是哪年哪月入职这家公司的？"),
        EvidenceSlot("employment_end", "离职/被辞退日期（精确到月）",
                     priority=SlotPriority.P0_CRITICAL, category="work",
                     ask_hint="你是哪年哪月离职/被辞退的？（如果还在职就说在职）"),
        EvidenceSlot("signed_contract", "是否签了书面劳动合同",
                     priority=SlotPriority.P0_CRITICAL, category="work",
                     ask_hint="公司有没有和你签书面劳动合同？"),

        # --- 工资 P0 ---
        EvidenceSlot("monthly_salary", "月平均应发工资（税前，含奖金补贴）",
                     priority=SlotPriority.P0_CRITICAL, category="salary",
                     ask_hint="你月工资大概多少钱？（说税前应发数，含绩效/年终奖折算）"),
        EvidenceSlot("salary_proof", "是否有工资流水/工资条作为证据",
                     priority=SlotPriority.P1_IMPORTANT, category="evidence",
                     ask_hint="你有银行工资流水或者公司发的工资条吗？"),

        # --- 解除详情 P0/P1 ---
        EvidenceSlot("termination_reason", "解除原因（谁提的/因为什么）",
                     priority=SlotPriority.P0_CRITICAL, category="termination",
                     ask_hint="解除劳动关系是谁提的？原因是什么？（如：公司说经营不善辞退 / 我主动提离职）"),
        EvidenceSlot("termination_form", "解除形式（口头/书面/微信群）",
                     priority=SlotPriority.P1_IMPORTANT, category="termination",
                     ask_hint="辞退/解除是口头说的、聊天记录里说的、还是书面通知书？有截图吗？"),
        EvidenceSlot("already_compensation", "公司已支付的补偿金额",
                     priority=SlotPriority.P1_IMPORTANT, category="termination",
                     ask_hint="公司已经给你任何补偿了吗？给了多少钱？"),

        # --- 常见赔偿请求关联槽 P1 ---
        EvidenceSlot("overtime_info", "加班情况（时长/类型/证据）",
                     priority=SlotPriority.P2_SUPPORT, category="salary",
                     ask_hint="有加班费诉求吗？大概多少小时，是平日/周末/法定节假日？有打卡记录吗？"),
        EvidenceSlot("annual_leave_left", "未休年休假天数",
                     priority=SlotPriority.P2_SUPPORT, category="salary",
                     ask_hint="今年还有未休的年假吗？大概多少天？"),
        EvidenceSlot("non_compete", "是否签过竞业限制协议",
                     priority=SlotPriority.P3_OPTIONAL, category="work",
                     ask_hint="你有和公司签过竞业限制协议吗？"),

        # --- 社保 P2 ---
        EvidenceSlot("social_insurance_status", "社保缴纳情况",
                     priority=SlotPriority.P2_SUPPORT, category="social_insurance",
                     ask_hint="公司有没有给你交社保？是按实际工资交还是最低基数？"),
    ]

    def __init__(self, slots: Optional[list[EvidenceSlot]] = None):
        # 用独立副本，避免污染 STANDARD_SLOTS 类变量
        self._slots: dict[str, EvidenceSlot] = {
            s.key: EvidenceSlot(**s.__dict__) for s in (slots or self.STANDARD_SLOTS)
        }
        self._history: list[SlotUpdateRecord] = []

    # ============================================================
    # 核心 API：槽位读写
    # ============================================================
    def update_slot(self, key: str, value: Any, *,
                    state: SlotState = SlotState.PROVIDED,
                    confidence: float = 1.0,
                    source: str = "user_message",
                    reason: str = "") -> SlotUpdateRecord:
        """更新单个槽位，返回 update 记录（可追溯）"""
        if key not in self._slots:
            # 允许动态加自定义槽（比如工伤类用户可以临时加工伤等级槽）
            self._slots[key] = EvidenceSlot(
                key=key, label=key, state=state, value=value,
                confidence=confidence, source=source,
            )
        slot = self._slots[key]
        record = SlotUpdateRecord(
            key=key,
            old_state=slot.state, new_state=state,
            old_value=slot.value, new_value=value,
            reason=reason,
        )
        slot.value = value
        slot.state = state
        slot.confidence = confidence
        slot.source = source
        self._history.append(record)
        return record

    def batch_update(self, updates: dict[str, Any], *,
                     default_state: SlotState = SlotState.DERIVED,
                     source: str = "llm_extract") -> list[SlotUpdateRecord]:
        """LLM 从一段自由文本里提取了一堆字段 → 批量 update"""
        records = []
        for k, v in updates.items():
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            records.append(self.update_slot(
                k, v, state=default_state, confidence=0.75, source=source,
                reason=f"从用户输入 LLM 抽取: {k}={v}",
            ))
        return records

    def get_slot(self, key: str) -> Optional[EvidenceSlot]:
        return self._slots.get(key)

    def list_slots(self, state: Optional[SlotState] = None,
                   priority: Optional[SlotPriority] = None,
                   category: Optional[str] = None) -> list[EvidenceSlot]:
        result = list(self._slots.values())
        if state:
            result = [s for s in result if s.state == state]
        if priority:
            result = [s for s in result if s.priority == priority]
        if category:
            result = [s for s in result if s.category == category]
        return result

    # ============================================================
    # 核心：追问推荐（每轮 Agent 问用户的 1-3 个问题从这来）
    # ============================================================
    def next_questions_to_ask(self, max_questions: int = 2) -> list[EvidenceSlot]:
        """
        返回这一轮应该追问的槽，排序算法：
            1. 先按 priority.value 升序（P0 先问）
            2. 同优先级按 confidence 升序（置信度低的先问）
            3. 同 confidence 按历史更新时间老的先问
        """
        # 只挑 MISSING 或 DERIVED（DERIVED 也需要确认一下）
        candidates = [s for s in self._slots.values()
                      if s.state in (SlotState.MISSING, SlotState.DERIVED)]
        candidates.sort(key=lambda s: (s.priority.value, s.confidence, s.key))
        return candidates[:max_questions]

    def completion_ratio(self) -> tuple[int, int, float]:
        """
        信息完整度统计：(已完成数, 总数, 完成率)
        面试讲：这是 FSM 判断「是否足够进入 ANALYZING 阶段」的量化阈值
        """
        # 只统计 P0+P1，P2/P3 不影响"能不能给结论"
        counted = [s for s in self._slots.values() if s.priority.value <= 1]
        # PROVIDED（用户明确提供）+ DERIVED（LLM 推断）都算「够分析」
        # 面试讲：这里是个经典的"精度 vs 进度"权衡。
        #   完全用 PROVIDED → 用户被问烦；完全用 DERIVED → 推断错了赔偿就错。
        #   折中：ANALYZING 阶段允许用 DERIVED 做粗算，但最终报告里会
        #         把所有 DERIVED 项列出来提醒用户「此为推断值请确认」。
        OK_STATES = {SlotState.PROVIDED, SlotState.DERIVED}
        completed = sum(1 for s in counted if s.state in OK_STATES and s.value is not None)
        return completed, len(counted), (completed / len(counted) if counted else 0.0)

    # ============================================================
    # BaseMemory 接口实现（让证据槽也能被统一 read/write 检查）
    # ============================================================
    def write(self, role: str, content, **metadata) -> MemoryItem:
        """统一接口的 write：当成一次 slot 变更事件写入 history trace"""
        item = MemoryItem(role=role, content=content, metadata=metadata,
                          timestamp=len(self._history))
        return item

    def read(self, limit=None, **filters) -> list[MemoryItem]:
        """把每个槽当前状态包装成 MemoryItem 返回"""
        items = [
            MemoryItem(
                role="slot_state",
                content={
                    "key": s.key, "label": s.label, "value": s.value,
                    "state": s.state.value, "confidence": s.confidence,
                    "priority": s.priority.name, "category": s.category,
                },
                timestamp=i,
            ) for i, s in enumerate(self._slots.values())
        ]
        if limit:
            items = items[-limit:]
        return items

    def clear(self) -> None:
        # 重置为标准槽位默认值（MISSING）
        self._slots = {s.key: EvidenceSlot(**s.__dict__) for s in self.STANDARD_SLOTS}
        self._history.clear()

    # ============================================================
    # 辅助：生成结构化 Dict（给 Tool / LLM Prompt 拼接用）
    # ============================================================
    def to_flat_dict(self, include_not_provided: bool = True) -> dict:
        """
        生成平坦 dict 给下游工具/LLM 使用
        :param include_not_provided: True = 包含 PROVIDED + DERIVED（推荐默认）
                                     False = 只包含用户明确 PROVIDED
        """
        OK_STATES = {SlotState.PROVIDED}
        if include_not_provided:
            OK_STATES.add(SlotState.DERIVED)
        return {
            s.key: s.value for s in self._slots.values()
            if s.state in OK_STATES and s.value is not None
        }

    def pretty_summary(self) -> str:
        """人类可读的证据槽摘要（调试/打印用）"""
        lines = ["📋 证据槽状态:"]
        done, total, ratio = self.completion_ratio()
        lines.append(f"   完整度: {done}/{total} 关键项 ({ratio:.0%})")
        for p in (SlotPriority.P0_CRITICAL, SlotPriority.P1_IMPORTANT,
                  SlotPriority.P2_SUPPORT, SlotPriority.P3_OPTIONAL):
            group = [s for s in self._slots.values() if s.priority == p]
            if not group:
                continue
            lines.append(f"   【{p.name}】")
            for s in group:
                mark = "✅" if s.state == SlotState.PROVIDED else (
                    "🤔" if s.state == SlotState.DERIVED else "❌")
                val_display = str(s.value) if s.value is not None else "(未填)"
                conf = f" conf={s.confidence:.2f}" if s.state == SlotState.DERIVED else ""
                lines.append(f"     {mark} {s.label}: {val_display}{conf}")
        return "\n".join(lines)