"""
src.state.orchestrator —— AgentOrchestrator（把 FSM + Memory + Tools 串起来的总驱动）

V3 升级：基于 LangGraph 的 StateGraph + Checkpoint 架构

核心变化：
    V1：handle_message() 根据 FSM 当前状态 → 硬编码调用 _state_*_flow()
    V2：handle_message() → _run_agent_loop() → LLM 自主决策每一步
    V3：handle_message() → LaborLawGraph.run() → LangGraph StateGraph + SQLite Checkpoint

V3 架构优势：
    1. 内置 Checkpoint：每次节点执行后自动保存状态快照（SQLite），进程重启可恢复
    2. 图结构天然支持复杂分支和循环
    3. 状态 Schema 由 TypedDict 定义，类型安全 + 自动序列化
    4. 支持流式输出（stream）和 Human-in-the-Loop（interrupt）
    5. 与 LangSmith/LangFuse 等可观测性工具无缝集成

💡 面试讲解点：
    · 「LangGraph 的 StateGraph 替代自研 FSM」——获得内置 Checkpoint + 流式 + 可观测性
    · 业务逻辑（extractor/report_gen/reviewer）完全复用，只替换了状态管理层
    · 图结构天然支持复杂分支和循环，比线性 FSM 更灵活
    · 规则兜底保留：问候语拦截、转出关键词（工伤/社保）仍用规则，不消耗 LLM 调用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .fsm import AgentState, DialogueFSM
from .session import AgentSession, SessionStore, get_session_store
from .tools import ToolRegistry, get_registry, MockLawLookUpTool, LawLookupTool
from .extractor import LLMEvidenceExtractor
from .report import ReportGenerator, ReportReviewer
from ..core.llm_client import get_llm_client
from ..core.logger import logger
from ..memory import SlotState


# ============================================================
# 配置常量：控制 FSM 跃迁的阈值（面试讲：这些是调参点）
# ============================================================
COMPLETION_RATIO_THRESHOLD = 0.65   # 证据槽 P0/P1 完整度超过这个比例 → 允许进入 ANALYZING
MAX_QUESTIONS_PER_TURN = 2          # 每轮最多追问用户几个问题
REFERRING_KEYWORDS = ("工伤认定", "伤残等级", "社保补缴", "公积金补缴",
                      "退休", "档案转移", "确认劳动关系超过1年")
# 完成度不足但最多追问的轮数 — 防止无限循环 COLLECTING
MAX_COLLECTING_ROUNDS = 5


# ============================================================
# Orchestrator 处理结果
# ============================================================
@dataclass
class AgentResponse:
    session_id: str
    current_state: AgentState
    response_text: str
    fsm_transition_reason: str = ""
    suggested_actions: list[str] = field(default_factory=list)
    debug_info: dict = field(default_factory=dict)

    def pretty(self, show_debug: bool = True) -> str:
        lines = [
            f"🤖 [State: {self.current_state.value}]",
            f"   {self.response_text}",
        ]
        if self.suggested_actions:
            lines.append("   💡 接下来你可以: " + " / ".join(self.suggested_actions))
        if show_debug and self.debug_info:
            import json
            lines.append("   🔍 Debug: " + json.dumps(self.debug_info, ensure_ascii=False))
        return "\n".join(lines)


# ============================================================
# Mock LLM 抽取器（演示用，不真调大模型）
# 面试时讲：真实环境下替换成 core.llm_client.LLMClient.run_prompt()
# ============================================================
def mock_llm_extract_slots(user_text: str, session: AgentSession) -> dict:
    """
    从用户自由文本里抽取证据槽字段。
    V1 用规则模拟，演示 FSM 链路；真实生产用 LLM Prompt + Function Calling 返回 JSON。
    """
    import re
    result: dict = {}
    text = user_text

    # --- 入职/离职日期 ---
    m = re.search(r"(20\d{2})[年\-/](\d{1,2})月?", text)
    if m and "入职" in text and "employment_start" not in session.evidence_slots.to_flat_dict(True):
        result["employment_start"] = f"{m.group(1)}-{int(m.group(2)):02d}"
    # 第二组日期：如果有两个年-月，第二个当离职日期
    dates = list(re.finditer(r"(20\d{2})[年\-/](\d{1,2})月?", text))
    if len(dates) >= 2:
        m2 = dates[-1]
        result["employment_end"] = f"{m2.group(1)}-{int(m2.group(2)):02d}"
    elif ("被辞退" in text or "离职" in text or "开除" in text or "开了" in text) and \
         "employment_end" not in session.evidence_slots.to_flat_dict(True):
        # 没有日期也至少触发槽状态更新
        pass

    # --- 工资 ---
    m = re.search(r"([一二三四五六七八九十百千0-9.]+)\s*(千|块|元|k|w|万)?\s*(一个月|每月|月薪|月工资|工资)", text, re.IGNORECASE)
    if m:
        num_str = m.group(1).replace("千", "1000").replace("万", "10000")
        try:
            salary_val = int(float(num_str))
            unit = m.group(2) or ""
            if unit in ("千", "k", "K"): salary_val *= 1000
            if unit in ("万", "w", "W"): salary_val *= 10000
            if 2000 <= salary_val <= 200000:
                result["monthly_salary"] = salary_val
        except ValueError:
            pass
    # 另一种模式："工资到手8500" / "税前工资12000" / "税前12000"
    # ✅ 关键修复：
    #   1) 上下文必须出现「工资类关键词」，避免补偿金数字被误判
    #   2) 用 finditer 遍历所有候选（因为 2021/6 这些年份数字会被先匹配），
    #      选数值范围合法 3000~150000 且最近距离工资关键词的那个
    has_salary_context = any(k in text for k in ("工资", "月薪", "薪资", "税前", "税后",
                                                 "应发", "实发", "收入", "到手"))
    if has_salary_context and "monthly_salary" not in result:
        p = re.compile(r"(到手|税前|应发|实发)?\s*(?:工资|月薪|薪资)?\s*"
                       r"([0-9]{3,6})\s*(?:工资|月薪|薪资)?\s*(元|块)?")
        best_val = None
        best_dist = 10 ** 9  # 离最近工资关键词的字符距离，越小越准
        salary_keyword_indices = [text.find(k) for k in ("工资", "税前", "税后", "月薪",
                                                         "薪资", "收入", "到手", "应发", "实发")
                                  if k in text]
        for m in p.finditer(text):
            try:
                v = int(m.group(2))
            except (ValueError, TypeError):
                continue
            if not (3000 <= v <= 150000):
                continue
            # ✅ 防误判：如果这个数字在补偿金语境（给了我/到账/补偿等）附近，就不算月薪
            lo = max(0, m.start(2) - 10)
            hi = min(len(text), m.end(2) + 8)
            ctx_win = text[lo:hi]
            if any(k in ctx_win for k in ("给我", "给了我", "给我的", "到账", "补偿",
                                          "遣散", "打给", "赔偿", "先付", "预付")):
                continue
            mid = m.start(2)
            d = min(abs(mid - i) for i in salary_keyword_indices)
            if d < best_dist:
                best_dist = d
                best_val = v
        if best_val is not None:
            result["monthly_salary"] = best_val

    # --- 合同/辞退形式/补偿/地区 ---
    if ("没签" in text or "未签" in text or "没有签" in text) and "合同" in text:
        result["signed_contract"] = False
    elif "签了合同" in text or "有合同" in text:
        result["signed_contract"] = True
    if ("北京" in text or "上海市" in text or "深圳" in text or "广州" in text or "杭州" in text):
        for city in ("北京", "深圳", "广州", "杭州"):
            if city in text:
                result["company_city"] = city
                break
        if "上海市" in text or "上海" in text:
            result["company_city"] = "上海"
    if "口头" in text or "微信" in text or "聊天记录" in text or "群里" in text:
        if "辞" in text or "开" in text or "离职" in text:
            result["termination_form"] = "口头/聊天记录（建议截图+录屏）"
    if ("书面" in text or "通知书" in text) and "辞" in text:
        result["termination_form"] = "书面解除通知书"
    if "公司" in text and ("经营不善" in text or "效益不好" in text or "裁员" in text):
        result["termination_reason"] = "公司以经营不善/经济性裁员为由解除"
    elif "老板" in text and ("开了" in text or "开除" in text or "辞退" in text):
        result["termination_reason"] = "口头辞退（疑似违法解除，待进一步确认）"
    elif "主动" in text and ("离职" in text or "辞职" in text):
        result["termination_reason"] = "劳动者主动提出辞职"
    m = re.search(r"给了我?\s*([0-9]+)\s*(千|万|元|块)?", text)
    already_comp_keywords = ("补偿", "遣散", "n+1", "n＋1", "赔偿", "到账",
                             "已经付", "已经给", "打了", "打给我", "给了我")
    if m and any(k in text.lower() or k in text for k in already_comp_keywords):
        try:
            val = int(m.group(1))
            unit = m.group(2) or "元"
            if unit in ("万", "w"): val *= 10000
            elif unit == "千": val *= 1000
            result["already_compensation"] = val
        except ValueError:
            pass
    # ✅ 修复：匹配「流水/工资条/转账记录」就认为有工资证明 —— 不用写死 "有流水"
    if "流水" in text or "工资条" in text or "转账记录" in text or "工资卡" in text:
        result["salary_proof"] = True
    return result


# ============================================================
# Mock LLM 报告生成器
# ============================================================
def mock_llm_generate_report(session: AgentSession, rag_text: str, tool_text: str) -> str:
    """把证据槽 + RAG 结果 + Tool 结果拼成一段自然语言的最终报告"""
    slots = session.evidence_slots
    flat = slots.to_flat_dict(True)
    done, total, ratio = slots.completion_ratio()
    sections = []
    sections.append(f"📋【案情摘要】已确认 {done}/{total} 关键项（完整度{ratio:.0%}）。")
    if flat.get("company_city"): sections.append(f"    · 地区: {flat['company_city']}")
    if flat.get("employment_start") and flat.get("employment_end"):
        sections.append(f"    · 在职时间: {flat['employment_start']} → {flat['employment_end']}")
    if flat.get("monthly_salary"): sections.append(f"    · 月工资（估）: {flat['monthly_salary']} 元")
    if flat.get("signed_contract") is False: sections.append("    · ⚠️ 未签书面劳动合同")
    if flat.get("termination_reason"): sections.append(f"    · 解除事由: {flat['termination_reason']}")
    if flat.get("already_compensation"): sections.append(f"    · 公司已支付补偿: {flat['already_compensation']} 元")
    sections.append("")
    if rag_text:
        sections.append(f"📜【相关法条】（检索摘要）\n{rag_text}")
        sections.append("")
    if tool_text:
        sections.append(f"🧮【工具输出】\n{tool_text}")
        sections.append("")
    sections.append("🏁【最终结论】")
    sections.append("   基于当前信息，建议：")
    sections.append("   ① 不要签署任何「自愿离职书」/「协商解除协议」文件；")
    sections.append("   ② 3 天内准备社保缴费记录、工资流水、辞退通知截图三件核心证据；")
    sections.append("   ③ 先与公司书面协商要求法定补偿（协商函模板我可以提供）；")
    sections.append("   ④ 协商不成 → 12333 劳动监察投诉 → 劳动仲裁立案（时效 1 年）。")
    return "\n".join(sections)


# ============================================================
# Mock LLM 反思校验器
# ============================================================
def mock_llm_reflection(report_text: str, session: AgentSession) -> tuple[bool, str]:
    """V1 简单规则校验：只要报告里包含了至少 2 个法条名 + 提到「证据」就通过"""
    passed = (("第" in report_text and "条" in report_text)
              and ("证据" in report_text)
              and len(report_text) > 200)
    reason = "通过：法条引用 + 证据建议均已包含" if passed else "失败：报告缺少法条或证据建议"
    return passed, reason


# ============================================================
# 核心：AgentOrchestrator
# ============================================================
class AgentOrchestrator:
    """
    使用方式：
        orch = AgentOrchestrator()
        resp = orch.handle_message(None, "老板把我开了怎么办？")
        → 返回 AgentResponse（含 session_id，下一次就把 session_id 传进来继续对话）
    """

    def __init__(self,
                 session_store: Optional[SessionStore] = None,
                 tool_registry: Optional[ToolRegistry] = None,
                 auto_register_default_tools: bool = True,
                 mode: str = "auto",
                 llm=None,
                 rag_agent=None,
                 checkpoint_db_path: str = "data/checkpoints.db"):
        """
        mode:
            "real"  全部使用真实组件（LLM 抽取 / RAG 检索 / LLM 报告 / 规则校验）
            "mock"  全部使用 Mock（离线演示，原 V1 行为）
            "auto"  优先真实组件，LLM/检索失败时自动降级到兜底实现（默认）
        checkpoint_db_path:
            LangGraph SQLite checkpoint 数据库路径
        """
        self.mode = mode
        self.rag_agent = rag_agent
        self.store = session_store or get_session_store()
        self.tools = tool_registry or get_registry()
        self._checkpoint_db_path = checkpoint_db_path

        # --- LLM：懒加载 ---
        self.llm = llm
        if mode != "mock" and self.llm is None:
            try:
                self.llm = get_llm_client()
            except Exception as e:
                logger.warning(f"[Orchestrator] LLM 初始化失败，降级 mock 模式: {e}")
                self.llm = None
                self.mode = "mock"

        # --- 真实组件 ---
        real_enabled = self.mode != "mock"
        self.extractor = LLMEvidenceExtractor(llm=self.llm, enabled=real_enabled)
        self.report_generator = ReportGenerator(llm=self.llm, rag_agent=rag_agent, enabled=real_enabled)
        self.reviewer = ReportReviewer(enabled=real_enabled)

        # --- 工具注册 ---
        if auto_register_default_tools and not self.tools.has("law_lookup"):
            if self.mode == "mock":
                self.tools.register(MockLawLookUpTool())
            else:
                self.tools.register(LawLookupTool(rag_agent=rag_agent))

        # 计数器：用于 MAX_COLLECTING_ROUNDS 兜底
        self._collect_rounds: dict[str, int] = {}

        # V3：创建 LangGraph 图（替代 V2 的 Agent Loop）
        self._graph = None
        self._init_graph()

    def _init_graph(self):
        """初始化 LangGraph 图（延迟创建，依赖 llm/tools/extractor 等已就绪）"""
        try:
            from .graph import LaborLawGraph
            self._graph = LaborLawGraph(
                extractor=self.extractor,
                report_generator=self.report_generator,
                reviewer=self.reviewer,
                tools=self.tools,
                llm=self.llm,
                db_path=self._checkpoint_db_path if self.mode != "mock" else None,
                max_collect_rounds=MAX_COLLECTING_ROUNDS,
                completion_threshold=COMPLETION_RATIO_THRESHOLD,
            )
            logger.info(f"[Orchestrator] LangGraph 图初始化成功 "
                        f"(checkpoint={'SQLite' if self.mode != 'mock' else 'Memory'})")
        except Exception as e:
            logger.warning(f"[Orchestrator] LangGraph 图初始化失败，降级 V2 Agent Loop: {e}")
            self._graph = None

    # ============================================================
    # 对外主入口（V3：LangGraph StateGraph + Checkpoint）
    # ============================================================
    def handle_message(self,
                       session_id: Optional[str],
                       user_input: str,
                       **user_meta) -> AgentResponse:
        """每次用户发消息，调这一个方法就够了"""
        user_input = (user_input or "").strip()
        if not user_input:
            return AgentResponse(
                session_id=session_id or "new",
                current_state=AgentState.INIT,
                response_text="你好！我可以帮你分析劳动纠纷。请简单描述一下你的情况（入职时间、工资情况、遇到的问题），我会一步步给你结论和行动方案。",
            )

        # --- 拿/建 Session（仅用于 session_id 管理，LangGraph 独立管理状态） ---
        sess, created = self.store.get_or_create(session_id, **user_meta)
        if created:
            self._collect_rounds[sess.session_id] = 0

        # --- 特殊：用户明确要"重新开始" ---
        if user_input in ("reset", "重新开始", "重来"):
            sess.reset_all()
            self._collect_rounds[sess.session_id] = 0
            return AgentResponse(
                session_id=sess.session_id,
                current_state=sess.fsm.current_state,
                response_text="好的，已清空所有案情。请重新描述你需要咨询的问题。",
                fsm_transition_reason="用户主动 reset",
                debug_info={"cleared": True},
            )

        # --- 转出前置拦截：工伤/社保补缴仍用规则 ---
        if any(kw in user_input for kw in REFERRING_KEYWORDS):
            return self._do_refer_out(sess, user_input)

        # --- V3：LangGraph 图驱动（带 Checkpoint） ---
        if self._graph is not None and self.mode != "mock":
            return self._handle_via_graph(sess, user_input)

        # --- 降级：graph 不可用时走 mock 规则 ---
        return self._handle_mock_fallback(sess, user_input)

    def _handle_via_graph(self, sess: AgentSession, user_input: str) -> AgentResponse:
        """通过 LangGraph 图处理请求"""
        result = self._graph.run(
            session_id=sess.session_id,
            user_input=user_input,
            user_meta=sess.user_meta,
        )

        response_text = result.get("response_text", "")
        phase = result.get("current_phase", "init")

        # 映射 phase 到 AgentState
        phase_map = {
            "init": AgentState.INIT,
            "understanding": AgentState.UNDERSTANDING,
            "collecting": AgentState.COLLECTING_EVIDENCE,
            "analyzing": AgentState.ANALYZING,
            "generating": AgentState.GENERATING,
            "reviewing": AgentState.REVIEWING,
            "finalizing": AgentState.FINALIZING,
            "referring_out": AgentState.REFERRING_OUT,
        }
        current_state = phase_map.get(phase, AgentState.INIT)

        return AgentResponse(
            session_id=sess.session_id,
            current_state=current_state,
            response_text=response_text,
            fsm_transition_reason=f"LangGraph: phase={phase}",
            suggested_actions=result.get("suggested_actions", []),
            debug_info={
                "graph_phase": phase,
                "evidence_slots": result.get("evidence_slots", {}),
                "collect_rounds": result.get("collect_rounds", 0),
            },
        )

    def _handle_mock_fallback(self, sess: AgentSession, user_input: str) -> AgentResponse:
        """LangGraph 不可用时的 mock 降级处理"""
        done, total, ratio = sess.evidence_slots.completion_ratio()
        response_text = (
            f"（离线模式）已收到你的问题。当前信息完整度：{done}/{total}（{ratio:.0%}）。\n"
            "由于系统暂未连接 LLM，请补充更多信息或稍后重试。"
        )
        return AgentResponse(
            session_id=sess.session_id,
            current_state=sess.fsm.current_state,
            response_text=response_text,
            fsm_transition_reason="mock fallback",
            debug_info={"mode": "mock", "completion": f"{done}/{total}"},
        )

    # ============================================================
    # 状态快照查询（调试/监控用）
    # ============================================================
    def get_state_snapshot(self, session_id: str) -> Optional[dict]:
        """获取某个会话的当前状态快照"""
        if self._graph is not None:
            return self._graph.get_state_snapshot(session_id)
        return None

    def get_state_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """获取某个会话的状态历史（checkpoint 列表）"""
        if self._graph is not None:
            return self._graph.get_state_history(session_id, limit)
        return []

    def _do_refer_out(self, sess: AgentSession, user_input: str) -> AgentResponse:
        """工伤/社保补缴场景不适合 Agent 直接算 → 指引用户先走行政程序"""
        sess.fsm.transition(
            AgentState.REFERRING_OUT,
            reason=f"命中转出关键词: {[k for k in REFERRING_KEYWORDS if k in user_input]}",
            trigger="orchestrator",
        )
        matched_kw = next((k for k in REFERRING_KEYWORDS if k in user_input), "")
        guidance_map = {
            "工伤认定": "   📌 工伤赔偿必须先走行政程序：1个月内要求公司向参保地人社局提交工伤认定申请 → 拿到工伤认定决定书 → 再做劳动能力等级鉴定（1-10级）→ 有了等级后回来我帮你算三金（一次性伤残/医疗/就业补助金）",
            "社保补缴": "   📌 社保/公积金补缴不属于劳动仲裁受案范围，请直接向当地社保稽核部门 / 公积金管理中心投诉（12333 / 支付宝/微信 城市服务），或向税务部门（社保费已划转税务征收）举报。",
            "伤残等级": "   📌 伤残等级/劳动能力鉴定必须由人社局劳动能力鉴定委员会做出，个人不能自己评。流程：先做工伤认定 → 申请劳动能力鉴定 → 拿到等级结论后回来，我可以帮你计算全部赔偿。",
        }
        guidance = next((v for k, v in guidance_map.items() if k in user_input),
                        "   📌 你的诉求涉及前置行政程序，请先向对应的行政部门申请认定/处理，拿到决定文书后再回我这里。")
        response = (
            f"⚠️ 你提到了「{matched_kw or '特殊诉求'}」—— 这类诉求我不能直接给赔偿数字，"
            "必须先走对应的行政/司法前置程序。\n"
            + guidance + "\n\n"
            "   前置程序走完后，你可以把认定书/决定书的关键信息发给我，我再帮你计算赔偿金额、写仲裁申请书。"
        )
        sess.short_term.write("assistant", response)
        sess.summary.write("assistant", response)
        return AgentResponse(
            sess.session_id, AgentState.REFERRING_OUT,
            response,
            fsm_transition_reason=f"命中转出关键词: {matched_kw}",
            suggested_actions=["补充新信息后回来继续", "说「重新开始」"],
            debug_info={"refer_keyword": matched_kw},
        )

    # ============================================================
    # 内部辅助
    # ============================================================
    @staticmethod
    def _extract_keyword_for_tool(user_input: str) -> str:
        for kw in ("辞退", "加班费", "未签合同", "年假", "拖欠工资"):
            if kw in user_input:
                return kw
        return "辞退"

    @staticmethod
    def _build_rag_query(sess: AgentSession, user_input: str) -> str:
        """把证据槽已确认的关键信息拼进检索 query，提升多轮检索相关性"""
        flat = sess.evidence_slots.to_flat_dict(True)
        if not flat:
            return user_input
        parts = []
        if flat.get("employment_start"):
            parts.append(f"入职{flat['employment_start']}")
        if flat.get("employment_end"):
            parts.append(f"离职{flat['employment_end']}")
        if flat.get("monthly_salary"):
            parts.append(f"月薪{flat['monthly_salary']}元")
        if flat.get("signed_contract") is False:
            parts.append("未签劳动合同")
        if flat.get("termination_reason"):
            parts.append(str(flat["termination_reason"]))
        if flat.get("company_city"):
            parts.append(f"地区{flat['company_city']}")
        suffix = "；".join(parts)
        return f"{user_input}；已确认案情：{suffix}"

    @staticmethod
    def _read_tool_cache(sess: AgentSession):
        """从短期记忆取出最新 ToolResult（read_tool_cache 返回 MemoryItem，需解包 .content）"""
        item = sess.short_term.read_tool_cache("law_lookup")
        return item.content if item is not None else None

    # ============================================================
    # 资源释放
    # ============================================================
    def close(self):
        """关闭编排器持有的共享资源（如延迟初始化的 RAG Agent）"""
        tool = self.tools.get("law_lookup")
        if tool is not None and hasattr(tool, "close"):
            try:
                tool.close()
            except Exception as e:
                logger.warning(f"[Orchestrator] 关闭 law_lookup 工具失败: {e}")