"""
src.state.tools.rag_lookup —— 真实 RAG 法条/案例检索工具

把真实的六阶段 RAG 检索（LaborLawAgent.search）封装成 BaseTool：
    - 输入：用户原始描述（可选带证据槽增强后的 query）
    - 输出：法条上下文 + 案例上下文 + 逐条命中明细

设计要点：
    · 工具持有共享的 rag_agent（由 AgentOrchestrator 注入），避免重复加载模型
    · 支持注入 search_fn，便于离线单测（不加载模型/LLM）
    · 检索失败时 success=False，Orchestrator 会降级为无检索生成
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ...core.logger import logger
from .base import BaseTool


def _format_results(results) -> list[dict]:
    """把 RankResult 列表转成可读 dict（无 agent 时的兜底格式）"""
    out = []
    for r in results[:10]:
        out.append({
            "chunk_id": r.chunk_id,
            "source_db": getattr(r, "source_db", ""),
            "law_name": getattr(r, "law_name", "") or getattr(r, "title", ""),
            "article": getattr(r, "article", ""),
            "score": round(float(getattr(r, "score", 0.0)), 4),
            "rank": getattr(r, "rank", 0),
            "text": (getattr(r, "text", "") or "")[:150],
        })
    return out


class LawLookupTool(BaseTool):
    """真实 RAG 检索工具：输入案情描述 → 返回相关法条与相似案例"""

    name = "law_lookup"
    description = ("输入一段劳动纠纷案情描述（或关键问题），返回检索到的相关法律条文与相似案例，"
                   "用于支撑赔偿计算与结论。")
    input_schema = {
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "用户案情描述或检索问题"},
            "top_k": {"type": "integer", "description": "返回多少条结果（默认 10）"},
        },
    }

    def __init__(self, rag_agent=None, search_fn: Optional[Callable[[str], dict]] = None):
        super().__init__()
        self._rag_agent = rag_agent
        self._search_fn = search_fn
        self._lazy_agent = None

    # ---------- 内部：获取检索能力 ----------
    def _get_search_fn(self) -> Callable[[str], dict]:
        if self._search_fn is not None:
            return self._search_fn
        if self._rag_agent is not None:
            return self._rag_agent.search
        if self._lazy_agent is None:
            # 延迟导入 + 延迟初始化，避免模块导入时加载模型
            from ...agent.orchestrator import LaborLawAgent
            logger.info("[LawLookupTool] 首次使用，初始化 LaborLawAgent（加载模型，约 10-20s）...")
            self._lazy_agent = LaborLawAgent()
        return self._lazy_agent.search

    @property
    def is_real(self) -> bool:
        return True

    def close(self):
        """释放延迟初始化的 RAG Agent（外部注入的由外部负责）"""
        if self._lazy_agent is not None:
            try:
                self._lazy_agent.close()
            except Exception as e:
                logger.warning(f"[LawLookupTool] 关闭 RAG Agent 失败: {e}")
            self._lazy_agent = None

    # ---------- 工具主逻辑 ----------
    def _run(self, params: dict) -> dict:
        query = (params.get("query") or "").strip()
        top_k = int(params.get("top_k") or 10)
        if not query:
            raise ValueError("query 不能为空")

        t0 = time.perf_counter()
        search_fn = self._get_search_fn()
        search_result = search_fn(query)
        elapsed = (time.perf_counter() - t0) * 1000

        results = search_result.get("results", []) or []
        checks = search_result.get("checks", {}) or {}

        # 优先用 agent 的上下文构建（含父块展开），否则用简易格式
        agent = self._rag_agent or getattr(self, "_lazy_agent", None)
        laws_ctx, cases_ctx = "", ""
        if agent is not None and hasattr(agent, "_build_context_split"):
            laws_ctx, cases_ctx = agent._build_context_split(results, checks)
        else:
            laws_ctx = "\n".join(
                f"### 【{r.law_name or r.title} {r.article}】score={r.score:.2f}\n{(r.text or '')[:500]}"
                for r in results if getattr(r, "source_db", "laws") == "laws"
            ) or "（未检索到相关法律依据）"
            cases_ctx = "\n".join(
                f"### 【案例 {r.law_name or r.title}】score={r.score:.2f}\n{(r.text or '')[:500]}"
                for r in results if getattr(r, "source_db", "laws") != "laws"
            )

        conf = checks.get("confidence")
        coverage = checks.get("coverage")
        logger.info(
            f"[LawLookupTool] query={query[:40]}... 命中={len(results)} 耗时={elapsed:.0f}ms"
        )
        return {
            "query": query,
            "n_results": len(results),
            "laws_context": laws_ctx,
            "cases_context": cases_ctx,
            "results": _format_results(results),
            "checks": {
                "confidence_ok": not (conf and getattr(conf, "should_abstain", False)),
                "coverage_ok": not (coverage and getattr(coverage, "needs_retry", False)),
                "retry_rounds": search_result.get("retry_rounds", 0),
            },
            "latency_ms": round(elapsed, 1),
        }
