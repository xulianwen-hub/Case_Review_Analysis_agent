"""
src.prompts —— 提示词层
    templates.py — 8 大分析模块的 Prompt 模板 + build_analysis_prompt()
"""
from .templates import (  # noqa: F401
    SYSTEM_PROMPT,
    MODULE_SUMMARY, MODULE_COMPENSATION, MODULE_LEGAL_BASIS,
    MODULE_CASES, MODULE_EVIDENCE, MODULE_ACTION_PLAN,
    MODULE_RISK, MODULE_CONCLUSION,
    build_analysis_prompt,
)