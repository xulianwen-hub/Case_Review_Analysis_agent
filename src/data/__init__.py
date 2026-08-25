"""
src.data —— 数据处理层（入库 / 管线编排）
    ingest.py   — 法条/案例入库逻辑 IngestManager / ingest_law()
    pipeline.py — 全流程管道编排器 PipelineRunner
"""
from .ingest import ingest_law, IngestManager  # noqa: F401
from .pipeline import PipelineRunner  # noqa: F401