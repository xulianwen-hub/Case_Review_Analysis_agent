"""
src.agent —— Agent 编排层（对外入口）
    orchestrator.py — LaborLawAgent 六阶段 RAG 主类
    api.py          — FastAPI HTTP 接口（/analyze  /ingest  /pipeline 等）
"""
from .orchestrator import LaborLawAgent, analyze_case  # noqa: F401