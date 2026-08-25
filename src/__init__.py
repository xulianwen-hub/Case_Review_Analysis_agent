"""
src —— 劳动纠纷分析 Agent 运行时核心包（分层结构）

包结构（src/ 顶层仅 7 个子包，清爽无杂）：
    src.core     — 基础设施：config / logger / llm_client（零业务依赖，最底层）
    src.rag      — RAG 检索链路：router / ranker / checker（Dense+BM25+RRF+CE 六阶段管线）
    src.prompts  — 提示词层：templates（8 大分析模块的模板）
    src.agent    — Agent 编排层：LaborLawAgent + api（对外入口）
    src.data     — 数据处理层：ingest / pipeline（入库/构建管线）
    src.memory   — 🧠 记忆 & 上下文管理模块：ShortTerm / EvidenceSlot / SummaryBuffer
    src.state    — 🔀 状态管理 & 工具编排模块：FSM / SessionStore / AgentOrchestrator / Tools

向后兼容说明：
    scripts/sandbox 里使用的旧路径（from src.config import X 等）通过
    sys.modules 别名机制自动重定向到对应子包，无需在 src/ 顶层保留 .py 转发文件。
    这是 numpy / pandas 等标准库广泛使用的官方兼容模式。

对外暴露的主要接口：
    from src import LaborLawAgent, get_llm_client
    from src import analyze_case
    from src import ingest_law, PipelineRunner
"""
import sys as _sys

# ============================================================
# 第一阶段：加载所有真实子包（触发各模块的初始化）
# ============================================================
# core 基础设施
from .core import config as _real_config
from .core import logger as _real_logger          # loguru logger 实例需要尽早 setup
from .core import llm_client as _real_llm_client

# rag 检索链路（Dense + BM25 并行 + RRF 粗排 + CrossEncoder 精排，无加权融合）
from .rag import router as _real_router
from .rag import ranker as _real_ranker
from .rag import checker as _real_checker

# prompts / agent / data
from .prompts import templates as _real_prompts
from .data import ingest as _real_ingest
from .data import pipeline as _real_pipeline

# ============================================================
# 第二阶段：sys.modules 路径别名（旧顶层路径 → 真实子包）
#
# 原理：Python 导入时先查 sys.modules 缓存；只要提前把真实模块
#       注入到 "src.config" 这个 key，后续任何代码
#       `from src.config import XXX` 都会直接命中，
#       不会再去文件系统寻找 src/config.py。
# ============================================================
_PREFIX = __name__ + "."  # 即 "src."

_sys.modules.setdefault(_PREFIX + "config",     _real_config)
_sys.modules.setdefault(_PREFIX + "logger",     _real_logger)
_sys.modules.setdefault(_PREFIX + "llm_client", _real_llm_client)

_sys.modules.setdefault(_PREFIX + "router",     _real_router)
_sys.modules.setdefault(_PREFIX + "ranker",     _real_ranker)
_sys.modules.setdefault(_PREFIX + "checker",    _real_checker)

# 【注意】src.prompts / src.agent 本身就是目录子包（有自己的 __init__.py）
# 不需要 sys.modules 别名！Python 原生就会 import 到包对象，且包的 __init__ 已导出符号。
# from src.prompts import X → 走 src/prompts/__init__.py re-export
# from src.agent import X   → 走 src/agent/__init__.py re-export

_sys.modules.setdefault(_PREFIX + "ingest",     _real_ingest)
_sys.modules.setdefault(_PREFIX + "pipeline",   _real_pipeline)

# ============================================================
# 第三阶段：顶层对外导出（"from src import X" 语法糖）
# ============================================================
from .agent import LaborLawAgent, analyze_case
from .core.llm_client import get_llm_client, LLMClientWithFallback
from .data.ingest import ingest_law, IngestManager
from .data.pipeline import PipelineRunner

# 🧠 Memory：4 个核心
from .memory import (
    BaseMemory, MemoryItem,
    ShortTermMemory, EvidenceSlotMemory, SummaryBuffer,
)

# 🔀 State + Orchestrator：4 个核心
from .state import (
    AgentState, DialogueFSM,
    AgentSession, SessionStore, AgentOrchestrator, AgentResponse,
)

# 🛠️ Tool：4 个核心（Tools 也在 state 子包里，也单独从 state 再导出方便记忆）
from .state import (
    BaseTool, ToolResult, ToolRegistry, MockLawLookUpTool, LawLookupTool,
    LLMEvidenceExtractor, ReportGenerator, ReportReviewer,
)

__all__ = [
    # LaborLawAgent + 管线（老接口保持）
    "LaborLawAgent",
    "get_llm_client",
    "LLMClientWithFallback",
    "analyze_case",
    "ingest_law",
    "IngestManager",
    "PipelineRunner",
    # Memory
    "BaseMemory", "MemoryItem",
    "ShortTermMemory", "EvidenceSlotMemory", "SummaryBuffer",
    # State / FSM
    "AgentState", "DialogueFSM",
    "AgentSession", "SessionStore",
    "AgentOrchestrator", "AgentResponse",
    # Tools
    "BaseTool", "ToolResult", "ToolRegistry", "MockLawLookUpTool", "LawLookupTool",
    # 多轮编排组件
    "LLMEvidenceExtractor", "ReportGenerator", "ReportReviewer",
]
