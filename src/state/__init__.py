"""
src.state —— 状态管理 & 编排模块（秋招面试终极核心）

子模块 & 面试要点：
    fsm                AgentState 9 状态 + DialogueFSM 有限状态机（跃迁钩子+非法跃迁拦截）
                       → 为什么选 FSM 而不是 ReAct / StateGraph？（业务刚性 + 可解释）

    session            AgentSession（FSM + 3 Memory 绑定 session_id）+ SessionStore
                       → 多用户并发 + TTL 过期 + 替换持久化（内存→Redis/SQLite）无需改业务代码

    orchestrator       AgentOrchestrator（总驱动：V3 LangGraph StateGraph + SQLite Checkpoint）
                       → V1 规则驱动 → V2 LLM Agent Loop → V3 LangGraph 图驱动
                       → 内置 Checkpoint：进程重启可恢复，天然支持流式输出

    graph              LaborLawGraph（LangGraph StateGraph 定义，企业级状态管理核心）
                       → StateGraph + Checkpointer（SQLite/Memory）替代自研 FSM
                       → 图结构天然支持复杂分支和循环，比线性 FSM 更灵活
                       → 业务逻辑（extractor/report_gen/reviewer）完全复用，只替换状态管理层

    tools              Tool 系统：
        base.py        BaseTool / ToolResult（参数校验+计时+异常捕获+function schema 导出）
        registry.py    ToolRegistry 单例注册中心（开闭原则）
        mock_law.py    Mock 法条工具（演示链路可用，替换为赔偿金/文档生成器 0 侵入）
        rag_lookup.py  真实 RAG 检索工具（包装 LaborLawAgent.search）

    extractor          LLMEvidenceExtractor：真实 LLM 结构化抽取证据槽（JSON + 正则兜底）
    report             ReportGenerator（8 模块报告）+ ReportReviewer（一致性校验）
"""
from .fsm import AgentState, DialogueFSM, TransitionEvent  # noqa: F401
from .session import (  # noqa: F401
    AgentSession, SessionStore, get_session_store,
)
from .orchestrator import AgentOrchestrator, AgentResponse  # noqa: F401
from .graph import LaborLawGraph, create_default_graph  # noqa: F401
from .tools import (  # noqa: F401
    BaseTool, ToolResult, ToolRegistry, get_registry, MockLawLookUpTool, LawLookupTool,
)
from .extractor import LLMEvidenceExtractor, parse_json_response  # noqa: F401
from .report import ReportGenerator, ReportReviewer, ReviewResult  # noqa: F401

__all__ = [
    # FSM
    "AgentState", "DialogueFSM", "TransitionEvent",
    # Session
    "AgentSession", "SessionStore", "get_session_store",
    # Orchestrator
    "AgentOrchestrator", "AgentResponse",
    # Tools
    "BaseTool", "ToolResult", "ToolRegistry", "get_registry", "MockLawLookUpTool", "LawLookupTool",
    # 抽取 / 报告
    "LLMEvidenceExtractor", "parse_json_response",
    "ReportGenerator", "ReportReviewer", "ReviewResult",
]