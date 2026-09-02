
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src.state.extractor import (  # noqa: E402
    parse_json_response,
    normalize_extracted,
    regex_extract_slots,
    LLMEvidenceExtractor,
)
from src.state.report import ReportReviewer, ReviewResult  # noqa: E402
from src.state.tools import LawLookupTool  # noqa: E402
from src.state.tools.registry import ToolRegistry  # noqa: E402
from src.state import AgentOrchestrator, AgentState  # noqa: E402
from src.rag.ranker import RankResult  # noqa: E402


# ============================================================
# 1. 证据抽取器
# ============================================================
class TestEvidenceExtractor:
    def test_parse_json_with_fence(self):
        text = '好的，结果如下：\n```json\n{"monthly_salary": 12000, "signed_contract": false}\n```'
        assert parse_json_response(text) == {"monthly_salary": 12000, "signed_contract": False}

    def test_parse_json_with_junk(self):
        text = '根据描述：{"company_city": "北京"} 以上就是结果'
        assert parse_json_response(text) == {"company_city": "北京"}

    def test_parse_json_invalid(self):
        assert parse_json_response("没有任何 JSON") is None
        assert parse_json_response("{broken") is None

    def test_normalize_money_and_bool(self):
        out = normalize_extracted({
            "monthly_salary": "1.2万",
            "already_compensation": "5000元",
            "signed_contract": "未签",
            "salary_proof": "有",
            "company_city": "北京",
        })
        assert out["monthly_salary"] == 12000
        assert out["already_compensation"] == 5000
        assert out["signed_contract"] is False
        assert out["salary_proof"] is True

    def test_regex_fallback(self):
        out = regex_extract_slots("我是2021年6月入职，税前工资12000，一直没签合同，老板口头辞退，给了5000块补偿")
        assert out["employment_start"] == "2021-06"
        assert out["monthly_salary"] == 12000
        assert out["signed_contract"] is False
        assert "辞退" in out["termination_reason"]
        assert out["already_compensation"] == 5000

    def test_llm_extractor_with_fake_llm(self, fake_llm):
        from src.memory import EvidenceSlotMemory
        slots = EvidenceSlotMemory()
        extractor = LLMEvidenceExtractor(llm=fake_llm, enabled=True)
        out = extractor.extract("我2021年6月入职，月薪12000，没签合同", slots)
        assert out["monthly_salary"] == 12000
        assert out["signed_contract"] is False

    def test_llm_extractor_fallback_on_garbage(self):
        from src.memory import EvidenceSlotMemory

        class GarbageLLM:
            def chat(self, messages, **kwargs):
                return "抱歉，我无法理解"

        slots = EvidenceSlotMemory()
        extractor = LLMEvidenceExtractor(llm=GarbageLLM(), enabled=True)
        out = extractor.extract("老板把我开了，给了我5000块补偿", slots)
        assert out.get("already_compensation") == 5000


# ============================================================
# 2. 报告反思校验器
# ============================================================
class TestReportReviewer:
    def _good_report(self) -> str:
        return (
            "## 案情摘要\n根据您的描述，公司口头辞退您，疑似违法解除劳动合同。\n\n"
            "## 法律依据\n《劳动合同法》第八十七条：公司违法解除劳动合同，"
            "应当依照第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金；"
            "未签订书面劳动合同可主张二倍工资差额（第八十二条）。\n\n"
            "## 证据清单\n请收集工资流水、解除通知截图、微信聊天记录等证据，"
            "其中工资流水和解除通知最为关键。\n\n"
            "## 行动路径建议\n先与公司书面协商，协商不成可向劳动监察投诉"
            "或申请劳动仲裁，注意仲裁时效为一年。\n\n"
            "## 法律定性结论\n公司的行为大概率构成违法解除劳动合同。\n\n"
            "> ⚠️ **免责声明**：以上分析仅供参考，不构成法律意见。"
        )

    def test_review_pass(self, session_with_slots):
        review = ReportReviewer().review(self._good_report(), session_with_slots)
        assert review.passed is True
        assert isinstance(review, ReviewResult)

    def test_review_fail_missing_sections(self, session_with_slots):
        bad = (
            "您好，根据您的描述，我大致了解了您的情况。您在公司工作了几年，"
            "最近被公司要求离开岗位，双方就离职补偿问题产生了分歧，"
            "您希望得到一些帮助和建议。这里我先简单回应一下，具体内容还需要进一步沟通确认，"
            "您可以补充更多细节，我再给您更完整的回复。另外，从您的描述来看，"
            "目前双方的分歧主要集中在离职原因和补偿标准两个方面，"
            "这直接影响到后续能够主张的金额和程序选择，建议您把入职时间、"
            "月工资标准、有没有签订书面合同、离职时公司出具了什么文件等信息整理清楚，"
            "方便后续进一步沟通时能够快速定位问题所在。"
        )
        review = ReportReviewer().review(bad, session_with_slots)
        assert review.passed is False
        assert any("免责声明" in i for i in review.issues)
        assert any("法条" in i for i in review.issues)

    def test_review_slot_consistency(self, session_with_slots):
        # 槽里是违法辞退 + 未签合同，但报告没提赔偿金/二倍工资 → 应失败
        report = (
            "## 案情摘要\n老板把我开了，公司位于北京，我在公司工作了五年多，"
            "月工资一万二，一直没有签订书面劳动合同，也没有缴纳社保。\n\n"
            "## 法律依据\n《劳动法》第三条：劳动者享有平等就业和选择职业的权利。\n\n"
            "## 证据清单\n工资流水、考勤记录、工作群聊天记录。\n\n"
            "## 行动路径建议\n先协商，再申请劳动仲裁，最后考虑诉讼。\n\n"
            "## 法律定性结论\n公司解除行为存在违法嫌疑，建议尽快维权。\n\n"
            "> ⚠️ **免责声明**：仅供参考，不构成法律意见。"
        )
        review = ReportReviewer().review(report, session_with_slots)
        assert review.passed is False
        assert any("赔偿金" in i for i in review.issues)
        assert any("二倍工资" in i for i in review.issues)


# ============================================================
# 3. Orchestrator 全流程（假 LLM + 假检索）
# ============================================================
class FakeLLM:
    """假的 LLM：抽取时返回 JSON；生成报告时返回预设文本"""

    def __init__(self, report_text: str):
        self.report_text = report_text

    def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        user = messages[1]["content"]
        if "抽取" in system and "槽位" in user:
            if "2021年6月" in user:
                return (
                    '{"company_city": "北京", "employment_start": "2021-06", '
                    '"monthly_salary": 12000, "signed_contract": false, '
                    '"termination_reason": "老板口头辞退（违法解除）", "salary_proof": true}'
                )
            return '{"termination_reason": "老板口头辞退（违法解除）"}'
        return self.report_text


def fake_search(query: str) -> dict:
    law = RankResult(
        chunk_id="law_87", text="《劳动合同法》第八十七条 违法解除支付二倍赔偿金。",
        law_name="劳动合同法", article="第八十七条", chapter="", parent_key="",
        score=0.9, rank=1, source="dense", source_db="laws", title="",
    )
    case = RankResult(
        chunk_id="case_1", text="【案例】某公司口头辞退员工被判违法解除赔偿。",
        law_name="某公司口头辞退案", article="", chapter="", parent_key="",
        score=0.8, rank=2, source="dense", source_db="non_laws", title="某公司口头辞退案",
    )
    return {"results": [law, case], "routing": None, "query_set": {},
            "checks": {"confidence": None, "coverage": None}, "retry_rounds": 0}


def build_orchestrator(report_text: str) -> AgentOrchestrator:
    registry = ToolRegistry()
    registry.register(LawLookupTool(search_fn=fake_search))
    return AgentOrchestrator(
        mode="real",
        llm=FakeLLM(report_text),
        tool_registry=registry,
        auto_register_default_tools=False,
    )


class TestOrchestratorRealFlow:
    def test_full_fsm_flow(self):
        report = (
            "## 案情摘要\n根据您的描述，公司口头辞退您，疑似违法解除。\n\n"
            "## 法律依据\n《劳动合同法》第八十七条：违法解除应支付赔偿金（2倍经济补偿）；"
            "未签书面合同可主张二倍工资差额。\n\n"
            "## 证据清单\n请收集工资流水、解除通知截图、聊天记录等证据。\n\n"
            "## 行动路径建议\n先协商，协商不成向劳动监察投诉或申请劳动仲裁，注意一年时效。\n\n"
            "> ⚠️ **免责声明**：以上分析仅供参考，不构成法律意见。"
        )
        orch = build_orchestrator(report)
        sid = None

        # 第 1 轮：信息足够 → 一次走完 ANALYZING→TOOL_USE→GENERATING→REVIEWING→FINALIZING
        r1 = orch.handle_message(sid, "老板把我开了，我是2021年6月入职，月薪12000，一直没签合同，在北京")
        assert r1.current_state == AgentState.FINALIZING
        assert "《劳动合同法》第八十七条" in r1.response_text

        # 证据槽被真实抽取写入
        sess = orch.store.get(r1.session_id)
        flat = sess.evidence_slots.to_flat_dict(True)
        assert flat["monthly_salary"] == 12000
        assert flat["signed_contract"] is False
        assert flat["company_city"] == "北京"
        assert flat["termination_reason"] == "老板口头辞退（违法解除）"

        # FSM 轨迹包含 TOOL_USE（真实检索工具）
        states = [e.from_state for e in sess.fsm.transitions]
        assert AgentState.TOOL_USE in states

        # 第 2 轮：重新开始 → 全部重置
        r2 = orch.handle_message(r1.session_id, "重新开始")
        assert r2.current_state == AgentState.INIT
        sess2 = orch.store.get(r1.session_id)
        assert sess2.evidence_slots.completion_ratio()[0] == 0

    def test_collecting_then_analyzing(self):
        report = (
            "## 案情摘要\n\n## 法律依据\n《劳动合同法》第八十七条 违法解除赔偿金。\n\n"
            "## 证据清单\n工资流水。\n\n## 行动路径\n申请劳动仲裁。\n\n"
            "> ⚠️ **免责声明**：仅供参考，不构成法律意见。"
        )
        orch = build_orchestrator(report)
        # 第 1 轮：信息不足 → COLLECTING 追问
        r1 = orch.handle_message(None, "老板把我开了")
        assert r1.current_state == AgentState.COLLECTING_EVIDENCE
        assert "我还需要确认" in r1.response_text
        sid = r1.session_id

        # 第 2 轮：补充关键信息 → 直接到 FINALIZING
        r2 = orch.handle_message(sid, "我是2021年6月入职，月薪12000，没签合同，公司在北京")
        assert r2.current_state == AgentState.FINALIZING


# ============================================================
# 4. mock 模式回归（原离线 demo 不受影响）
# ============================================================
class TestOrchestratorMockMode:
    def test_mock_mode_still_works(self):
        orch = AgentOrchestrator(mode="mock")
        r1 = orch.handle_message(None, "老板把我开了，给了5000块")
        # mock 正则只能抽到很少字段 → 进入追问
        assert r1.current_state == AgentState.COLLECTING_EVIDENCE
        assert orch.tools.has("law_lookup")
        orch.close()


# ============================================================
# fixtures
# ============================================================
@pytest.fixture
def fake_llm():
    class _F:
        def chat(self, messages, **kwargs):
            return '{"monthly_salary": 12000, "signed_contract": false}'
    return _F()


@pytest.fixture
def session_with_slots():
    from src.state.session import SessionStore
    store = SessionStore()
    sess, _ = store.get_or_create(None)
    sess.evidence_slots.batch_update(
        {"monthly_salary": 12000, "termination_reason": "老板口头辞退（违法解除）", "signed_contract": False},
        default_state=__import__("src.memory", fromlist=["SlotState"]).SlotState.PROVIDED,
    )
    return sess


@pytest.fixture(autouse=True)
def _fresh_tool_registry():
    """ToolRegistry 是单例，每个测试前重置，避免工具互相覆盖"""
    ToolRegistry._instance = None
    yield
    ToolRegistry._instance = None
