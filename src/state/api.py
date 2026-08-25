"""
src.state.api —— 多轮对话 FastAPI 路由（session 化）

接口：
    POST /api/agent/chat       —— 发一条消息，按 FSM 状态推进并返回（带 session_id）
    POST /api/agent/reset      —— 重置指定 session
    GET  /api/agent/sessions   —— 当前活跃 session 数

与 /api/chat 的区别：
    /api/chat           单轮：每次独立跑完整 RAG + 8 模块报告（无状态）
    /api/agent/chat     多轮：同一 session 内 FSM 推进，证据槽累积，追问 → 分析闭环

启动：uvicorn src.agent.api:app --port 8000
"""
from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import AgentOrchestrator
from ..core.logger import logger


state_router = APIRouter(prefix="/api/agent", tags=["agent-multiturn"])


# ============================================================
# 单例 Orchestrator（进程内共享；mode 可用环境变量 AGENT_MODE 覆盖）
# ============================================================
_orch: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orch
    if _orch is None:
        mode = os.getenv("AGENT_MODE", "auto")
        logger.info(f"[多轮对话] 初始化 AgentOrchestrator（mode={mode}）...")
        _orch = AgentOrchestrator(mode=mode)
    return _orch


# ============================================================
# 请求 / 响应模型
# ============================================================
class AgentChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(default=None, description="会话 ID（新会话传空）")


class AgentChatResponse(BaseModel):
    session_id: str
    state: str
    response_text: str
    fsm_transition_reason: str = ""
    suggested_actions: list = []
    slots_summary: dict = {}
    debug: dict = {}
    elapsed_seconds: float = 0.0


class AgentResetRequest(BaseModel):
    session_id: str = Field(..., description="要重置的会话 ID")


class AgentSessionsResponse(BaseModel):
    total_sessions: int
    expired_cleaned: int = 0


def _slots_summary(orch: AgentOrchestrator, session_id: str) -> dict:
    sess = orch.store.get(session_id)
    if sess is None:
        return {}
    done, total, ratio = sess.evidence_slots.completion_ratio()
    return {
        "completion": f"{done}/{total}",
        "ratio": round(ratio, 2),
        "slots": sess.evidence_slots.to_flat_dict(True),
    }


# ============================================================
# 接口
# ============================================================
@state_router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest):
    orch = get_orchestrator()
    start = time.time()
    try:
        resp = orch.handle_message(req.session_id, req.message)
    except Exception as e:
        logger.error(f"[多轮对话] 处理失败: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {e}")
    return AgentChatResponse(
        session_id=resp.session_id,
        state=resp.current_state.value,
        response_text=resp.response_text,
        fsm_transition_reason=resp.fsm_transition_reason,
        suggested_actions=resp.suggested_actions,
        slots_summary=_slots_summary(orch, resp.session_id),
        debug=resp.debug_info,
        elapsed_seconds=round(time.time() - start, 2),
    )


@state_router.post("/reset")
async def agent_reset(req: AgentResetRequest):
    orch = get_orchestrator()
    sess = orch.store.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"会话不存在或已过期: {req.session_id}")
    sess.reset_all()
    return {"status": "ok", "session_id": req.session_id, "state": sess.fsm.current_state.value}


@state_router.get("/sessions", response_model=AgentSessionsResponse)
async def agent_sessions():
    orch = get_orchestrator()
    cleaned = orch.store.cleanup_expired()
    return AgentSessionsResponse(
        total_sessions=orch.store.total_sessions(),
        expired_cleaned=cleaned,
    )
