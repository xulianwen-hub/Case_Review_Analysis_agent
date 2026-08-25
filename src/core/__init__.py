"""
src.core —— 基础设施层（零业务依赖，最底层）
    config.py     — 全局配置
    logger.py     — 日志封装（loguru）
    llm_client.py — LLM 多模型降级客户端
"""
from .config import *  # noqa: F401, F403
from .logger import logger  # noqa: F401
from .llm_client import (  # noqa: F401
    PROVIDER_CONFIG, CircuitState, CircuitBreaker,
    LLMClient, LLMClientWithFallback, get_llm_client,
)