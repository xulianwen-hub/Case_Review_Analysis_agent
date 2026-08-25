"""
src.rag —— RAG 检索链路层
    router.py    — 阶段一/二：意图路由、查询改写、HyDE
    ranker.py    — 阶段三/四/五：多路召回、RRF 融合、Cross-Encoder 精排
    checker.py   — 阶段六：置信度 / 覆盖度 / 冲突 三重自检
"""
from .router import IntentRouter, QueryRewriter, HyDERewriter, QueryVariant  # noqa: F401
from .ranker import (  # noqa: F401
    MultiPathRetriever, RRFMerger, CrossEncoderRanker,
    RetrievedDoc, RankResult, DBSource, get_cache,
)
from .checker import (  # noqa: F401
    ConfidenceReport, ConfidenceChecker,
    CoverageReport, CoverageChecker,
    ConflictReport, ConflictDetector,
    run_checks,
)