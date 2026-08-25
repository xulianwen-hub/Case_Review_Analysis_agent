"""
阶段六：Agent 上下文构建与再决策（三重自检）

检查一：置信度检查 — CE Top-1 分数 + Top-1 vs Top-3 差距
检查二：覆盖度检查 — 子问题覆盖（仅 Complex 时触发）
检查三：冲突检测 — 版本冲突标注
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..core.config import (
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
    CONFIDENCE_GAP_THRESHOLD, CONFIDENCE_GAP_CAUTIOUS,
    COVERAGE_LIGHTWEIGHT_ENABLED, COVERAGE_SUPPLEMENT_RETRIEVAL_MAX,
)
from ..core.logger import logger


# ============================================================
# 数据类
# ============================================================

@dataclass
class ConfidenceReport:
    """置信度检查报告"""
    level: str          # "high" / "medium" / "low"
    top1_score: float
    top3_gap: float     # Top-1 与 Top-3 分数差距
    message: str        # 提示信息
    disclaimer: str     # 免责声明建议
    should_abstain: bool  # 是否应该反问/拒绝回答


@dataclass
class CoverageReport:
    """覆盖度检查报告"""
    all_covered: bool
    missing_sub_questions: List[str] = field(default_factory=list)
    need_supplement: bool = False


@dataclass
class ConflictReport:
    """冲突检测报告"""
    has_conflict: bool
    conflict_items: List[str] = field(default_factory=list)
    prompt_annotation: str = ""  # 附加到 Prompt 的标注


# ============================================================
# 检查一：置信度检查
# ============================================================

class ConfidenceChecker:
    """
    检查一：置信度检查

    Top-1 CE 分数：
      > 0.7  → 高置信度，直接回答
      0.4-0.7 → 正常回答 + 免责声明
      < 0.4 → 触发反问

    补充检查：Top-1 与 Top-3 差距
      > 0.3 → 答案明确
      < 0.1 → 多个答案皆有可能
    """

    @staticmethod
    def check(reranked_results: List, ce_scores_available: bool = True) -> ConfidenceReport:
        """
        执行置信度检查

        Args:
            reranked_results: 精排后的结果列表（每个有 .score 属性）
            ce_scores_available: CE 分数是否可用（数据少时可能跳过 CE）

        Returns:
            ConfidenceReport
        """
        if not reranked_results:
            return ConfidenceReport(
                level="low",
                top1_score=0.0,
                top3_gap=0.0,
                message="未检索到任何相关文档",
                disclaimer="",
                should_abstain=False,
            )

        top1_score = reranked_results[0].score if ce_scores_available else 0.5

        # 计算 Top-3 分数差距
        if len(reranked_results) >= 3:
            top3_scores = [r.score for r in reranked_results[:3]]
            top3_gap = top3_scores[0] - top3_scores[-1]
        elif len(reranked_results) >= 2:
            top3_gap = reranked_results[0].score - reranked_results[-1].score
        else:
            top3_gap = 0.0

        # 判断置信度等级
        if ce_scores_available:
            if top1_score > CONFIDENCE_HIGH:
                level = "high"
                message = "检索结果高度匹配"
                disclaimer = ""
                should_abstain = False
            elif top1_score >= CONFIDENCE_MEDIUM:
                level = "medium"
                message = "检索结果中等匹配"
                disclaimer = "以上分析仅供参考，建议咨询专业律师。"
                should_abstain = False
            else:
                level = "low"
                message = "知识库中未找到与您问题高度匹配的信息"
                disclaimer = ""
                should_abstain = True
        else:
            # 无 CE 分数时仅基于 gap 判断
            level = "medium"
            message = "基于混合检索结果（未使用 Cross-Encoder 精排）"
            disclaimer = "以上分析仅供参考，建议咨询专业律师。"
            should_abstain = False

        # 补充检查：分数差距
        if not should_abstain and len(reranked_results) >= 3 and ce_scores_available:
            if top3_gap > CONFIDENCE_GAP_THRESHOLD:
                message += "，答案明确度较高"
            elif top3_gap < CONFIDENCE_GAP_CAUTIOUS:
                message += "，多个可能答案接近，需更谨慎"
                if level == "high":
                    level = "medium"

        logger.info(
            f"[置信度] level={level} | top1={top1_score:.4f} | "
            f"gap={top3_gap:.4f} | should_abstain={should_abstain}"
        )

        return ConfidenceReport(
            level=level,
            top1_score=top1_score,
            top3_gap=top3_gap,
            message=message,
            disclaimer=disclaimer,
            should_abstain=should_abstain,
        )


# ============================================================
# 检查二：覆盖度检查
# ============================================================

class CoverageChecker:
    """
    检查二：覆盖度检查（仅 Complex 时触发）

    第一步：轻量检查（关键词匹配，<5ms）
    第二步：LLM 深度检查（关键词失败时）

    补充检索：最多 1 轮
    """

    @staticmethod
    def lightweight_check(sub_questions: List[str],
                          top_k_texts: List[str]) -> CoverageReport:
        """
        轻量覆盖度检查（关键词匹配）

        Args:
            sub_questions: 子问题列表
            top_k_texts: Top-K 检索结果的文本列表

        Returns:
            CoverageReport
        """
        if not COVERAGE_LIGHTWEIGHT_ENABLED or not sub_questions:
            return CoverageReport(all_covered=True)

        missing = []
        combined_text = " ".join(top_k_texts)

        for sq in sub_questions:
            # 提取子问题的核心关键词（2-3字）
            keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', sq)
            core_kw = keywords[:3] if keywords else [sq[:4]]

            # 检查是否至少有一个关键词出现在 Top-K 文本中
            found = any(kw in combined_text for kw in core_kw)
            if not found:
                missing.append(sq)

        all_covered = len(missing) == 0
        need_supplement = not all_covered and COVERAGE_SUPPLEMENT_RETRIEVAL_MAX > 0

        if missing:
            logger.info(f"[覆盖度] 缺失子问题 ({len(missing)}/{len(sub_questions)}): {missing}")

        return CoverageReport(
            all_covered=all_covered,
            missing_sub_questions=missing,
            need_supplement=need_supplement,
        )

    @staticmethod
    def deep_check(sub_questions: List[str],
                   top_k_texts: List[str],
                   llm_client) -> CoverageReport:
        """
        LLM 深度覆盖度检查（仅轻量失败时触发）

        使用 LLMClientWithFallback 的降级机制：
        优先 DeepSeek → 失败自动切换 doubao → qwen
        断路器保证单次崩溃后不再重试同厂商

        Args:
            sub_questions: 子问题列表
            top_k_texts: Top-K 检索结果文本
            llm_client: LLM 客户端（LLMClientWithFallback 实例）

        Returns:
            CoverageReport
        """
        doc_text = "\n\n---\n\n".join(top_k_texts[:10])
        sq_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))

        prompt = f"""你是一个法律问题覆盖度检查器。判断以下子问题是否能从提供的法律文档中找到答案。

子问题列表：
{sq_text}

检索到的法律文档：
{doc_text[:3000]}

输出 JSON 格式（只输出 JSON）：
{{
    "all_covered": true/false,
    "missing": ["缺失的子问题1", ...]
}}

如果所有子问题都能在文档中找到至少部分相关信息，all_covered=true。"""

        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            used_provider = getattr(llm_client, 'provider', 'unknown')
            logger.info(f"[覆盖度] LLM 深度检查完成，使用厂商: {used_provider}")

            import json
            match = re.search(r'\{[^}]+\}', response.strip())
            if match:
                data = json.loads(match.group())
                result = CoverageReport(
                    all_covered=data.get("all_covered", True),
                    missing_sub_questions=data.get("missing", []),
                    need_supplement=not data.get("all_covered", True),
                )
                if result.need_supplement:
                    logger.info(f"[覆盖度] 深度检查确认缺失: {result.missing_sub_questions}")
                return result

        except RuntimeError as e:
            # 所有厂商均失败（降级链路耗尽）
            logger.error(f"[覆盖度] LLM 深度检查失败（所有厂商不可用）: {e}")
        except Exception as e:
            logger.warning(f"[覆盖度] LLM 深度检查异常: {e}")

        # 降级：LLM 不可用时，以轻量检查结果为准（不阻塞管线）
        return CoverageReport(
            all_covered=False,
            missing_sub_questions=sub_questions,
            need_supplement=False,
        )


# ============================================================
# 检查三：冲突检测
# ============================================================

class ConflictDetector:
    """
    检查三：冲突检测

    第一步：规则检测（同一条文的不同版本）
    第二步：在 Prompt 中标注冲突，由 LLM 在生成阶段解决

    设计原则：只标注，不自动解决
    """

    @staticmethod
    def detect(top_k_results: List) -> ConflictReport:
        """
        检测版本冲突

        Args:
            top_k_results: 排序后的结果列表（每个有 law_name, article, text 等属性）

        Returns:
            ConflictReport
        """
        # 按 (article 关键词, chapter) 分组，检测是否有多个版本
        article_groups: Dict[str, List] = {}
        for r in top_k_results:
            # 提取法条号（如 "第四十七条~第四十八条"）
            article_key = getattr(r, 'article', '')
            law_name = getattr(r, 'law_name', '')
            key = f"{law_name}|{article_key}"
            if key not in article_groups:
                article_groups[key] = []
            article_groups[key].append(r)

        conflicts = []
        for key, items in article_groups.items():
            if len(items) >= 2:
                # 检查是否有版本/修订差异
                text_hashes = set()
                unique_texts = []
                for item in items:
                    text = getattr(item, 'text', '')
                    h = hash(text)
                    if h not in text_hashes:
                        text_hashes.add(h)
                        unique_texts.append(text)

                if len(unique_texts) >= 2:
                    # 检测年份差异（2008版 vs 2012修正版）
                    year_pattern = r'(20\d{2})'
                    years_in_texts = []
                    for t in unique_texts:
                        found = re.findall(year_pattern, t)
                        if found:
                            years_in_texts.append(found[0])

                    if len(set(years_in_texts)) >= 2:
                        article = getattr(items[0], 'article', '')
                        conflicts.append(
                            f"发现 {article} 涉及不同年份版本: {', '.join(sorted(set(years_in_texts)))}"
                        )
                    else:
                        article = getattr(items[0], 'article', '')
                        conflicts.append(
                            f"发现 {article} 可能涉及不同版本/修订（内容有差异）"
                        )

        has_conflict = len(conflicts) > 0

        # 生成 Prompt 标注
        annotation = ""
        if has_conflict:
            annotation = (
                "⚠️ **注意**：检索结果中存在以下版本/内容冲突，请根据法律效力层级"
                "（新法优于旧法、上位法优于下位法）进行判断：\n"
            )
            for c in conflicts:
                annotation += f"- {c}\n"

        if has_conflict:
            logger.info(f"[冲突检测] 发现 {len(conflicts)} 个冲突项")

        return ConflictReport(
            has_conflict=has_conflict,
            conflict_items=conflicts,
            prompt_annotation=annotation,
        )


# ============================================================
# 检查工厂函数
# ============================================================

def run_checks(
    reranked_results: List,
    original_query: str,
    sub_questions: List[str] = None,
    llm_client=None,
    ce_scores_available: bool = True,
) -> Dict:
    """
    执行三重自检

    Args:
        reranked_results: 精排后的 Top-K 结果
        original_query: 原始查询
        sub_questions: 子问题列表（Complex 策略时）
        llm_client: LLM 客户端
        ce_scores_available: CE 分数是否可用

    Returns:
        {
            "confidence": ConfidenceReport,
            "coverage": CoverageReport,
            "conflict": ConflictReport,
            "needs_retry": bool,       # 是否需要补充检索
            "prompt_extra": str,       # 追加到 Prompt 的标注
        }
    """
    sub_questions = sub_questions or []

    # 检查一：置信度
    confidence = ConfidenceChecker.check(reranked_results, ce_scores_available)

    # 检查二：覆盖度（仅当有子问题时）
    coverage = CoverageReport(all_covered=True)
    if sub_questions and COVERAGE_LIGHTWEIGHT_ENABLED:
        top_k_texts = [getattr(r, 'text', '') for r in reranked_results[:10]]
        coverage = CoverageChecker.lightweight_check(sub_questions, top_k_texts)

        # 轻量检查失败 → 触发 LLM 深度检查（优先 DeepSeek，失败自动降级）
        if not coverage.all_covered and llm_client is not None:
            logger.info(
                f"[覆盖度] 轻量检查未通过，触发 LLM 深度检查 "
                f"(缺失 {len(coverage.missing_sub_questions)} 个子问题)"
            )
            deep_coverage = CoverageChecker.deep_check(
                sub_questions=sub_questions,
                top_k_texts=top_k_texts,
                llm_client=llm_client,
            )
            # 以深度检查结果为准
            coverage = deep_coverage

    # 检查三：冲突检测
    conflict = ConflictDetector.detect(reranked_results)

    # 汇总需要重新检索的条件
    needs_retry = coverage.need_supplement and not confidence.should_abstain

    # 构建 Prompt 附加内容
    prompt_extra = ""
    if confidence.disclaimer:
        prompt_extra += f"\n[置信度提示] {confidence.message}\n{confidence.disclaimer}\n"
    if conflict.prompt_annotation:
        prompt_extra += f"\n{conflict.prompt_annotation}\n"

    return {
        "confidence": confidence,
        "coverage": coverage,
        "conflict": conflict,
        "needs_retry": needs_retry,
        "prompt_extra": prompt_extra,
    }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=== 测试阶段六：置信度检查 ===")

    from dataclasses import dataclass

    @dataclass
    class MockResult:
        score: float
        text: str
        article: str
        law_name: str

    # 测试高置信度
    high_results = [
        MockResult(score=0.85, text="...", article="第四十七条", law_name="劳动合同法"),
        MockResult(score=0.72, text="...", article="第四十八条", law_name="劳动合同法"),
        MockResult(score=0.45, text="...", article="第八十七条", law_name="劳动合同法"),
    ]
    report = ConfidenceChecker.check(high_results)
    print(f"  高: level={report.level}, abstain={report.should_abstain}")

    # 测试低置信度
    low_results = [
        MockResult(score=0.35, text="...", article="第四十七条", law_name="劳动合同法"),
    ]
    report = ConfidenceChecker.check(low_results)
    print(f"  低: level={report.level}, abstain={report.should_abstain}")

    print("\n=== 测试覆盖度检查 ===")
    sub_qs = ["拖欠工资的赔偿标准", "解除劳动合同的条件"]
    texts = ["关于工资拖欠，用人单位应按劳动合同法规定支付经济补偿..."]
    cov = CoverageChecker.lightweight_check(sub_qs, texts)
    print(f"  all_covered: {cov.all_covered}, missing: {cov.missing_sub_questions}")

    print("\n=== 测试冲突检测 ===")
    conflict_results = [
        MockResult(score=0.9, text="2008年施行...", article="第四十七条", law_name="劳动合同法_2008版"),
        MockResult(score=0.8, text="2012年修正后施行...", article="第四十七条", law_name="劳动合同法_2012版"),
    ]
    conflict = ConflictDetector.detect(conflict_results)
    print(f"  has_conflict: {conflict.has_conflict}, items: {conflict.conflict_items}")