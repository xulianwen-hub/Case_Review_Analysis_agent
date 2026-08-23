"""
劳动纠纷案例分析 Agent — 六阶段 RAG 管线

阶段一：前置路由与意图解析（IntentRouter）
阶段二：查询改写策略选择（QueryRewriter）
阶段三：并行多路召回（MultiPathRetriever）
阶段四：RRF 融合与粗排（RRFMerger）
阶段五：Cross-Encoder 精排（CrossEncoderRanker，已启用 bge-reranker-base）
阶段六：Agent 上下文构建与再决策（三重自检）
"""
import sys
import os
import json
import time
import re
from typing import Optional, Dict, List

from ..core.config import (
    MIN_INPUT_LENGTH, MAX_INPUT_LENGTH,
    CHINESE_CHARS_PER_TOKEN, ENGLISH_CHARS_PER_TOKEN,
    MAX_CONTEXT_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS,
    CE_TOP_K_OUTPUT, CE_HYDE_THRESHOLD, CE_MAX_RETRY_ROUNDS,
    COVERAGE_SUPPLEMENT_RETRIEVAL_MAX,
)
from ..core.logger import logger
from ..core.llm_client import get_llm_client
from ..prompts.templates import SYSTEM_PROMPT, build_analysis_prompt

# 六阶段 RAG 组件
from ..rag.router import IntentRouter, QueryRewriter, HyDERewriter, QueryVariant
from ..rag.ranker import (
    MultiPathRetriever, RRFMerger, CrossEncoderRanker,
    RetrievedDoc, RankResult, get_cache,
)
from ..rag.checker import (
    ConfidenceChecker, CoverageChecker, ConflictDetector, run_checks,
)


class LaborLawAgent:
    """劳动纠纷智能分析 Agent（六阶段 RAG 管线）"""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("劳动纠纷案例分析 Agent 初始化中...")
        logger.info("=" * 60)

        # === LLM 客户端 ===
        logger.info("[1/4] 初始化 LLM 客户端...")
        self.llm = get_llm_client()
        logger.info(f"  LLM 模型: {self.llm.model}")
        logger.info(f"  API 地址: {self.llm.base_url}")

        # === 六阶段 RAG 组件 ===
        logger.info("[2/4] 初始化多路径检索器（多数据源）...")
        self.multi_retriever = MultiPathRetriever.from_config()

        logger.info("[3/4] 初始化 RRF + CE 排序器...")
        self.rrf_merger = RRFMerger()
        self.ce_ranker = CrossEncoderRanker()

        logger.info("[4/4] Agent 就绪!")
        logger.info("=" * 60)

    # ============================================================
    # 六阶段 RAG 管线
    # ============================================================

    def search(self, user_input: str) -> Dict:
        """
        完整的六阶段 RAG 检索管线

        Args:
            user_input: 用户原始输入

        Returns:
            {
                "results": List[RankResult],   # 最终精排结果
                "routing": RoutingDecision,     # 路由决策
                "query_set": Dict,              # 查询改写结果
                "checks": Dict,                 # 三重自检结果
                "retry_rounds": int,            # 重试轮次
            }
        """
        start_time = time.time()
        retry_rounds = 0

        # =========================================
        # 阶段一：前置路由与意图解析
        # =========================================
        routing = IntentRouter.route(user_input, llm_client=self.llm)

        if not routing.go_to_rag:
            return {
                "results": [],
                "routing": routing,
                "query_set": None,
                "checks": None,
                "retry_rounds": 0,
                "direct_reply": routing.direct_reply,
            }

        # =========================================
        # 阶段二：查询改写
        # =========================================
        query_set = QueryRewriter.rewrite(
            query=user_input,
            strategy=routing.strategy,
            llm_client=self.llm,
        )

        logger.info(
            f"[RAG管线] 策略={routing.strategy} | "
            f"候选Query数={len(query_set['queries'])} | "
            f"复杂度={routing.complexity:.2f}"
        )

        # =========================================
        # 阶段三+四+五：检索 + RRF + CE
        # =========================================
        reranked_results, retry_rounds = self._retrieve_and_rank(
            query_set=query_set,
            original_query=user_input,
            routing=routing,
        )

        # =========================================
        # 阶段六：三重自检
        # =========================================
        checks = run_checks(
            reranked_results=reranked_results,
            original_query=user_input,
            sub_questions=query_set.get("sub_questions", []),
            llm_client=self.llm,
            ce_scores_available=True,  # CE 已启用（bge-reranker-base）
        )

        # 覆盖度检查：轻量失败 → LLM 深度确认 → 补充检索
        # self.llm 是 LLMClientWithFallback，自带降级链路：
        #   DeepSeek（默认）→ 豆包 → 千问
        # 每个厂商有独立断路器，DeepSeek 失败一次自动切换
        if (checks["needs_retry"] and
            retry_rounds < CE_MAX_RETRY_ROUNDS and
            checks["coverage"].missing_sub_questions):

            missing_sqs = checks["coverage"].missing_sub_questions

            # 第二步：LLM 深度检查（优先 DeepSeek，崩溃一次自动切其他模型）
            logger.info(
                f"[覆盖度] 轻量检查失败（{len(missing_sqs)} 个缺失），"
                f"进入 LLM 深度检查（降级链路: {self.llm.fallback_chain}）..."
            )
            top_k_texts = [getattr(r, 'text', '') for r in reranked_results[:10]]

            deep_cov = CoverageChecker.deep_check(
                sub_questions=missing_sqs,
                top_k_texts=top_k_texts,
                llm_client=self.llm,
            )

            # 深度检查确认需要补充检索
            if deep_cov.need_supplement and deep_cov.missing_sub_questions:
                supplement_count = min(
                    len(deep_cov.missing_sub_questions),
                    COVERAGE_SUPPLEMENT_RETRIEVAL_MAX,
                )
                logger.info(
                    f"[覆盖度] 深度确认缺失 {len(deep_cov.missing_sub_questions)} 个子问题，"
                    f"补充检索 {supplement_count} 个"
                )

                for sq_text in deep_cov.missing_sub_questions[:supplement_count]:
                    query_set["queries"].append(QueryVariant(
                        text=sq_text,
                        source_type="sub_question_supplement",
                        weight=0.8,
                    ))

                reranked_results, _ = self._retrieve_and_rank(
                    query_set=query_set,
                    original_query=user_input,
                    routing=routing,
                    skip_cache=True,
                )
                retry_rounds += 1

                # 重新执行三重自检
                checks = run_checks(
                    reranked_results=reranked_results,
                    original_query=user_input,
                    sub_questions=query_set.get("sub_questions", []),
                    llm_client=self.llm,
                    ce_scores_available=True,
                )
            else:
                logger.info("[覆盖度] 深度检查认为子问题已覆盖，跳过补充检索")

        # HyDE 闭环：CE Top-1 < 0.4 且还有重试配额时触发
        if (checks["confidence"].should_abstain and
            retry_rounds < CE_MAX_RETRY_ROUNDS):
            logger.info("[HyDE] 置信度不足，触发 HyDE 二次检索...")
            hyde_doc, has_articles = HyDERewriter.generate_hypothetical_document(
                user_input, self.llm
            )
            if hyde_doc and not has_articles:
                # 将 HyDE 生成的假设文档作为额外 Query
                hyde_qv = type(query_set["queries"][0])(
                    text=hyde_doc, source_type="hyde", weight=0.7
                )
                query_set["queries"].append(hyde_qv)

                reranked_results, _ = self._retrieve_and_rank(
                    query_set=query_set,
                    original_query=user_input,
                    routing=routing,
                    skip_cache=True,  # HyDE 不走缓存
                )
                retry_rounds += 1

                # 重新执行三重自检
                checks = run_checks(
                    reranked_results=reranked_results,
                    original_query=user_input,
                    sub_questions=query_set.get("sub_questions", []),
                    llm_client=self.llm,
                    ce_scores_available=True,
                )

        total_time = time.time() - start_time
        logger.info(
            f"[RAG管线] 完成 | "
            f"结果数={len(reranked_results)} | "
            f"重试={retry_rounds} | "
            f"耗时={total_time:.2f}s"
        )

        return {
            "results": reranked_results,
            "routing": routing,
            "query_set": query_set,
            "checks": checks,
            "retry_rounds": retry_rounds,
        }

    def _retrieve_and_rank(
        self,
        query_set: Dict,
        original_query: str,
        routing,
        skip_cache: bool = False,
    ) -> tuple:
        """
        阶段三→四→五：多数据源检索 + 标准 RRF 粗排 + Cross-Encoder 精排

        Returns:
            (List[RankResult], retry_rounds)
        """
        # 阶段三：多数据源检索（Dense 和 BM25 独立返回）
        dense_docs, bm25_docs = self.multi_retriever.retrieve_all(
            queries=query_set["queries"],
            original_query=original_query,
        )

        logger.info(
            f"[多库召回] Dense={len(dense_docs)} | BM25={len(bm25_docs)} | "
            f"来源分布={self._count_by_source(dense_docs + bm25_docs)}"
        )

        # 阶段四：标准 RRF 跨数据源融合（Σ1/(k+rank)，不加权，保留 Dense/BM25 独立排名）
        rrf_results = self.rrf_merger.merge_multi_source(dense_docs, bm25_docs)

        logger.info(
            f"[RRF] 融合后={len(rrf_results)} | "
            f"Top-1 score={rrf_results[0].score:.4f}" if rrf_results else "[RRF] 无结果"
        )

        # 阶段五：Cross-Encoder 精排（bge-reranker-base，已启用）
        reranked_results = self.ce_ranker.rerank(
            query=original_query,
            candidates=rrf_results,
            top_k=CE_TOP_K_OUTPUT,
        )

        return reranked_results, 0

    @staticmethod
    def _count_by_source(docs: list) -> dict:
        """统计文档来源分布"""
        counts = {}
        for d in docs:
            src = getattr(d, 'source_db', 'unknown')
            counts[src] = counts.get(src, 0) + 1
        return counts

    # ============================================================
    # 上下文构建
    # ============================================================

    def _build_context(self, results: List[RankResult],
                       checks: Dict, max_texts: int = 10) -> str:
        """从检索结果构建 LLM 上下文（合并输出，向后兼容）"""
        laws_ctx, cases_ctx = self._build_context_split(results, checks, max_texts)
        if cases_ctx:
            return f"{laws_ctx}\n\n---\n\n{cases_ctx}"
        return laws_ctx

    def _build_context_split(self, results: List[RankResult], checks: Dict,
                             max_texts: int = 10) -> tuple:
        """
        从检索结果按 source_db 构建两个独立的上下文片段

        Returns:
            (laws_text: str, cases_text: str)
            - laws_text:  laws 库（法律法规）的上下文
            - cases_text: non_laws 库（案例/裁判/法规解读）的上下文；没有则为空串
        """
        if not results:
            return "（未检索到相关法律依据）", ""

        # 1. 按来源拆分成两组
        laws_results   = [r for r in results if getattr(r, 'source_db', 'laws') == 'laws']
        nonlaw_results = [r for r in results if getattr(r, 'source_db', 'laws') != 'laws']

        # 2. 组装法条部分
        laws_lines = []
        if checks and checks.get("prompt_extra"):
            laws_lines.append(checks["prompt_extra"])

        laws_count = 0
        seen_parents = set()
        per_group_max = max(1, max_texts // 2)

        for r in laws_results:
            if r.parent_key and r.parent_key not in seen_parents:
                seen_parents.add(r.parent_key)
                parent_text = self.multi_retriever.get_parent_text(r.parent_key, "laws")
                if parent_text:
                    laws_count += 1
                    laws_lines.append(
                        f"### 【{laws_count}】{r.law_name} {r.article} (相关性分数: {r.score:.2f})\n"
                        f"{parent_text}\n"
                    )
                    if laws_count >= per_group_max:
                        break
                    continue
            laws_count += 1
            laws_lines.append(
                f"### 【{laws_count}】{r.law_name} {r.article} (相关性分数: {r.score:.2f})\n"
                f"{r.text}\n"
            )
            if laws_count >= max_texts:
                break

        laws_text = "\n".join(laws_lines) if laws_lines else "（未检索到相关法律法规依据）"

        # 3. 组装案例/裁判部分
        cases_lines = []
        cases_count = 0
        seen_parents2 = set()

        for r in nonlaw_results:
            if r.parent_key and r.parent_key not in seen_parents2:
                seen_parents2.add(r.parent_key)
                parent_text = self.multi_retriever.get_parent_text(r.parent_key, "non_laws")
                if parent_text:
                    cases_count += 1
                    cases_lines.append(
                        f"### 【案例 {cases_count}】{r.law_name} {r.article} (相关性分数: {r.score:.2f})\n"
                        f"{parent_text}\n"
                    )
                    if cases_count >= per_group_max:
                        break
                    continue
            cases_count += 1
            cases_lines.append(
                f"### 【案例 {cases_count}】{r.law_name} {r.article} (相关性分数: {r.score:.2f})\n"
                f"{r.text}\n"
            )
            if cases_count >= max_texts:
                break

        cases_text = "\n".join(cases_lines) if cases_lines else ""
        return laws_text, cases_text

    # ============================================================
    # 分析接口
    # ============================================================

    def _validate_input(self, user_input: str) -> Optional[str]:
        if not user_input or not user_input.strip():
            return "输入不能为空"
        if len(user_input) < MIN_INPUT_LENGTH:
            return f"输入过短（至少 {MIN_INPUT_LENGTH} 个字符），请描述更多案情细节"
        if len(user_input) > MAX_INPUT_LENGTH:
            return f"输入过长（最多 {MAX_INPUT_LENGTH} 个字符），请精简描述"
        return None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / CHINESE_CHARS_PER_TOKEN + other_chars / ENGLISH_CHARS_PER_TOKEN)

    def analyze(self, user_input: str) -> str:
        """分析用户输入（含六阶段 RAG 管线）"""
        start_time = time.time()

        # 输入校验
        validation_error = self._validate_input(user_input)
        if validation_error:
            return f"⚠️ 输入校验失败: {validation_error}"

        logger.info("-" * 60)
        logger.info(f"[用户输入] {user_input[:200]}...")

        # === 六阶段 RAG 检索 ===
        search_result = self.search(user_input)

        # 非 RAG 场景（问候语）
        if search_result.get("direct_reply"):
            logger.info(f"[路由] 非RAG场景，直接回复")
            return search_result["direct_reply"]

        reranked_results = search_result["results"]
        checks = search_result.get("checks", {})

        # 低置信度 → 反问
        if checks and checks.get("confidence"):
            conf = checks["confidence"]
            if conf.should_abstain:
                return (
                    "⚠️ 抱歉，知识库中未找到与您问题高度匹配的法律依据。\n\n"
                    "建议您：\n"
                    "1. 提供更详细的案情描述（如具体的法条号、案由等）\n"
                    "2. 换个方式重新描述您遇到的问题\n"
                    "3. 拨打 12333 劳动保障热线或咨询专业律师"
                )

        # === 构建上下文 → LLM 生成 ===
        logger.info("[上下文] 构建检索上下文（法条+案例分离）...")
        laws_ctx, cases_ctx = self._build_context_split(
            results=reranked_results,
            checks=checks,
        )
        context_chars = len(laws_ctx) + len(cases_ctx)
        logger.info(f"  法条上下文: {len(laws_ctx)} 字符 | 案例上下文: {len(cases_ctx)} 字符 | 总计: {context_chars}")

        logger.info("[Prompt] 构建分析 Prompt...")
        user_message = build_analysis_prompt(user_input, laws_ctx, cases_ctx)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        total_tokens = self._estimate_tokens(
            sum(len(m["content"]) for m in messages) * "x"
        )
        logger.info(f"  Prompt: {total_tokens} tokens")

        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"  Token 数 ({total_tokens}) 超过安全上限 ({MAX_CONTEXT_TOKENS})")

        logger.info("[LLM] 调用 LLM 生成分析报告...")
        llm_start = time.time()
        try:
            response = self.llm.chat(
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except Exception as e:
            elapsed = time.time() - llm_start
            logger.error(f"[LLM] 调用失败 (耗时 {elapsed:.1f}s): {type(e).__name__}: {e}")
            return f"⚠️ 大模型调用异常，请稍后重试。错误: {e}"

        total_time = time.time() - start_time
        logger.info(
            f"[完成] LLM生成={time.time()-llm_start:.1f}s | "
            f"总耗时={total_time:.1f}s | "
            f"输出={len(response)}字符"
        )

        return response

    def analyze_stream(self, user_input: str):
        """流式分析"""
        validation_error = self._validate_input(user_input)
        if validation_error:
            yield f"⚠️ {validation_error}"
            return

        logger.info("-" * 60)
        logger.info(f"[用户输入] {user_input[:200]}...")

        # 六阶段 RAG 检索
        search_result = self.search(user_input)

        if search_result.get("direct_reply"):
            yield search_result["direct_reply"]
            return

        reranked_results = search_result["results"]
        checks = search_result.get("checks", {})

        if checks:
            conf = checks.get("confidence")
            if conf and conf.should_abstain:
                yield "⚠️ 抱歉，知识库中未找到与您问题高度匹配的法律依据。建议提供更详细的案情描述。"
                return

        laws_ctx, cases_ctx = self._build_context_split(reranked_results, checks)
        user_message = build_analysis_prompt(user_input, laws_ctx, cases_ctx)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        logger.info("[LLM] 流式生成中...")
        try:
            stream = self.llm.chat_stream(messages, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"[LLM] 流式调用失败: {type(e).__name__}: {e}")
            yield f"\n\n⚠️ 生成过程中断: {e}"

    def ingest_law(self, docx_path: str) -> Dict:
        """增量录入新法条，入库后自动清除检索缓存"""
        from .ingest import ingest_law as _ingest_law
        result = _ingest_law(docx_path)
        if result.get("status") == "ok":
            logger.info(f"[Agent] 法条录入成功，缓存已自动清除")
        return result

    def close(self):
        # self.retriever 是早期字段，当前实现统一使用 multi_retriever
        retriever = getattr(self, "retriever", None)
        if retriever is not None:
            retriever.close()
        self.multi_retriever.close()


# ============================================================
# 命令行入口
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("  劳动纠纷案例分析 Agent Demo (六阶段 RAG)")
    logger.info("  输入您的案情，Agent 给出专业分析")
    logger.info("  输入 'quit' 或 'exit' 退出")
    logger.info("=" * 60)

    agent = LaborLawAgent()

    try:
        while True:
            logger.info("-" * 60)
            user_input = input("\n请输入您的案情描述: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                logger.info("感谢使用！再见。")
                break

            report = agent.analyze(user_input)

            print("\n" + "=" * 60)
            print("分析报告")
            print("=" * 60)
            print(report)
            print("\n" + "=" * 60)

    except KeyboardInterrupt:
        logger.info("中断退出。")
    finally:
        agent.close()


def analyze_case(user_input: str) -> str:
    """一键分析函数"""
    agent = LaborLawAgent()
    try:
        return agent.analyze(user_input)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
