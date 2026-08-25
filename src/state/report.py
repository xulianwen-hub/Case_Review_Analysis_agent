"""
src.state.report —— 8 模块报告生成器 + 反思一致性校验器

GENERATING 状态：
    · 用真实的 SYSTEM_PROMPT + build_analysis_prompt（8 个模块模板）
    · 上下文来自 LawLookupTool 的真实检索结果（法条 + 案例分离）
    · LLM 不可用时降级为数据驱动的摘要报告（离线 demo 可用）

REVIEWING 状态：
    · 规则校验：免责声明 / 法条引用 / 证据建议 / 行动路径 / 与证据槽一致性
    · 失败返回具体 issue，Orchestrator 据此重新生成（最多重试 1 次）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.logger import logger
from ..prompts.templates import SYSTEM_PROMPT, build_analysis_prompt


# ============================================================
# 反思校验结果
# ============================================================
@dataclass
class ReviewResult:
    passed: bool
    reason: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# ============================================================
# 报告生成器
# ============================================================
class ReportGenerator:
    """用真实 LLM 生成 8 模块分析报告；失败时降级为摘要报告"""

    def __init__(self, llm: Any = None, rag_agent: Any = None, enabled: bool = True):
        self.llm = llm
        self.rag_agent = rag_agent
        self.enabled = enabled

    # ---------- 降级报告（离线可用，数据驱动） ----------
    @staticmethod
    def _fallback_report(session, tool_result: Optional[dict]) -> str:
        flat = session.evidence_slots.to_flat_dict(True)
        laws_ctx = (tool_result or {}).get("laws_context") or "（未检索到相关法律依据）"
        cases_ctx = (tool_result or {}).get("cases_context") or ""
        lines = [
            "## 案情摘要",
            "根据您的描述，我理解的情况是：",
        ]
        if flat.get("employment_start") or flat.get("employment_end"):
            lines.append(f"- 在职时间：{flat.get('employment_start', '未知')} → {flat.get('employment_end', '至今/未知')}")
        if flat.get("monthly_salary"):
            lines.append(f"- 月工资（估）：{flat['monthly_salary']} 元")
        if flat.get("signed_contract") is False:
            lines.append("- 未签书面劳动合同")
        if flat.get("termination_reason"):
            lines.append(f"- 解除事由：{flat['termination_reason']}")
        if flat.get("already_compensation"):
            lines.append(f"- 公司已支付补偿：{flat['already_compensation']} 元")
        lines.append("")
        lines.append("## 赔偿金额估算")
        lines.append("> ⚠️ 当前处于离线降级模式，未调用大模型精确计算；以下为按证据槽的初步提示：")
        if flat.get("monthly_salary"):
            lines.append(f"- 月工资：{flat['monthly_salary']} 元（请以实际应发工资为准）")
            reason = str(flat.get("termination_reason", ""))
            if any(k in reason for k in ("违法", "辞退", "开除", "开了")):
                lines.append("- 若构成违法解除：赔偿金 ≈ 月工资 × 工作年限 × 2（《劳动合同法》第87条）")
            else:
                lines.append("- 若属协商解除/无过失解除：经济补偿 N ≈ 月工资 × 工作年限（《劳动合同法》第47条）")
        else:
            lines.append("- 缺少月工资信息，无法估算金额，请补充。")
        lines.append("")
        lines.append("## 法律依据（检索结果摘要）")
        lines.append(laws_ctx[:1500])
        if cases_ctx:
            lines.append("")
            lines.append("## 相似案例参考（检索结果摘要）")
            lines.append(cases_ctx[:1000])
        lines.append("")
        lines.append("## 证据清单")
        lines.append("- 银行工资流水 / 工资条（证明工资标准与劳动关系）")
        if flat.get("signed_contract") is False:
            lines.append("- 未签合同：考勤、工牌、社保记录、工作群聊天（证明事实劳动关系）")
        lines.append("- 解除通知/聊天记录截图（证明解除原因与形式）")
        lines.append("")
        lines.append("## 行动路径建议")
        lines.append("1. 与公司书面协商（保留记录）→ 2. 12333 劳动监察投诉 → 3. 劳动仲裁（时效 1 年）→ 4. 诉讼")
        lines.append("")
        lines.append("## 法律定性结论")
        lines.append("基于当前信息，结论存在不确定性；建议补充完整信息后由大模型给出精确分析。")
        lines.append("")
        lines.append("> ⚠️ **免责声明**：以上分析仅供参考，不构成法律意见。建议咨询专业律师。")
        return "\n".join(lines)

    # ---------- 主入口 ----------
    def generate(self, user_input: str, session, tool_result: Optional[dict] = None) -> str:
        if not self.enabled or self.llm is None:
            return self._fallback_report(session, tool_result)

        laws_ctx = (tool_result or {}).get("laws_context") or "（未检索到相关法律依据）"
        cases_ctx = (tool_result or {}).get("cases_context") or ""
        user_message = build_analysis_prompt(user_input, laws_ctx, cases_ctx)
        # 多轮对话：把证据槽里已确认的案情汇总注入 Prompt，避免「忘记」前几轮信息
        case_summary = self._slots_summary_text(session)
        if case_summary:
            user_message = (
                "## 已确认的案情信息（来自多轮对话，请直接采用，不要再问）\n"
                f"{case_summary}\n\n---\n\n{user_message}"
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        try:
            logger.info("[报告生成] 调用 LLM 生成 8 模块报告...")
            report = self.llm.chat(
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
            return report
        except Exception as e:
            logger.warning(f"[报告生成] LLM 调用失败，降级摘要报告: {type(e).__name__}: {str(e)[:120]}")
            return self._fallback_report(session, tool_result)

    @staticmethod
    def _slots_summary_text(session) -> str:
        """把证据槽转成一行行可读的案情汇总"""
        flat = session.evidence_slots.to_flat_dict(True)
        if not flat:
            return ""
        lines = []
        if flat.get("employment_start") or flat.get("employment_end"):
            lines.append(f"- 在职时间：{flat.get('employment_start', '未知')} → {flat.get('employment_end', '至今/未知')}")
        if flat.get("company_city"):
            lines.append(f"- 公司城市：{flat['company_city']}")
        if flat.get("monthly_salary"):
            lines.append(f"- 月工资：{flat['monthly_salary']} 元")
        if flat.get("signed_contract") is False:
            lines.append("- 未签订书面劳动合同")
        if flat.get("termination_reason"):
            lines.append(f"- 解除原因：{flat['termination_reason']}")
        if flat.get("termination_form"):
            lines.append(f"- 解除形式：{flat['termination_form']}")
        if flat.get("already_compensation"):
            lines.append(f"- 公司已支付补偿：{flat['already_compensation']} 元")
        return "\n".join(lines)


# ============================================================
# 反思一致性校验器
# ============================================================
class ReportReviewer:
    """
    规则驱动的报告反思校验：
        passed=False 时给出 issues + suggestions，供 Orchestrator 重试。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def review(self, report: str, session, tool_result: Optional[dict] = None) -> ReviewResult:
        if not self.enabled:
            return ReviewResult(passed=True, reason="反思校验已禁用", issues=[], suggestions=[])
        if not report or len(report.strip()) < 200:
            return ReviewResult(
                passed=False,
                reason="报告过短，疑似生成失败",
                issues=["报告长度不足 200 字"],
                suggestions=["重新生成完整 8 模块报告"],
            )

        issues: List[str] = []
        suggestions: List[str] = []
        text = report

        # 1. 免责声明
        if not any(k in text for k in ("免责声明", "不构成法律意见", "仅供参考")):
            issues.append("缺少免责声明")
            suggestions.append("在报告末尾加入免责声明（不构成法律意见，建议咨询律师）")

        # 2. 法条引用
        law_re = re.compile(r"《[^》]+》|第\s*[0-9一二三四五六七八九十百]+\s*条")
        if not law_re.search(text):
            issues.append("缺少具体法条引用")
            suggestions.append("引用检索结果中的具体法条，如《劳动合同法》第47条")

        # 3. 证据建议
        if "证据" not in text:
            issues.append("缺少证据清单/证据建议")
            suggestions.append("补充证据清单模块（工资流水、解除通知、聊天记录等）")

        # 4. 行动路径
        if not any(k in text for k in ("仲裁", "协商", "投诉", "监察", "诉讼")):
            issues.append("缺少行动路径建议")
            suggestions.append("补充协商→投诉→仲裁→诉讼的行动路径与时效提醒")

        # 5. 与证据槽一致性（软检查：给出提示但不算硬性失败）
        flat = session.evidence_slots.to_flat_dict(True)
        reason = str(flat.get("termination_reason", ""))
        if any(k in reason for k in ("违法", "辞退", "开除", "开了")):
            if not any(k in text for k in ("赔偿金", "2倍", "二倍", "2N")):
                issues.append("报告未体现违法解除赔偿金（2N）")
                suggestions.append("根据解除事由明确违法解除赔偿金 = 经济补偿 × 2（第87条）")
        if flat.get("signed_contract") is False and "双倍" not in text and "二倍" not in text:
            issues.append("未签合同但报告未提及二倍工资主张")
            suggestions.append("补充未签书面劳动合同的二倍工资主张（最多11个月）")

        passed = not issues
        reason_text = "校验通过：法条 + 证据 + 行动路径 + 免责声明齐全" if passed else \
            f"校验未通过：{len(issues)} 项问题"
        return ReviewResult(
            passed=passed,
            reason=reason_text,
            issues=issues,
            suggestions=suggestions,
        )
