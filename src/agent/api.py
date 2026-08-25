"""
FastAPI 服务（支持 SSE 流式输出）

启动方式：
    python -m src.agent.api
    uvicorn src.agent.api:app --host 0.0.0.0 --port 8000 --reload

接口：
    POST /api/chat          —— 普通对话（一次性返回）
    POST /api/chat/stream   —— SSE 流式输出
    POST /api/agent/chat    —— 多轮对话（session 化，FSM 推进）
    GET  /api/health        —— 健康检查
    POST /api/ingest        —— 增量录入法条
"""
import json
import time
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .orchestrator import LaborLawAgent
from ..data.ingest import ingest_law
from ..data.pipeline import PipelineRunner
from ..core.logger import logger
from ..state.api import state_router

app = FastAPI(
    title="劳动纠纷智能分析 Agent",
    description="六阶段 RAG 管线 + SSE 流式输出",
    version="0.1.0",
)
app.include_router(state_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: Optional[LaborLawAgent] = None


def get_agent() -> LaborLawAgent:
    global _agent
    if _agent is None:
        _agent = LaborLawAgent()
    return _agent


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入", min_length=1, max_length=5000)
    stream: bool = Field(default=True, description="是否使用流式输出")


class ChatResponse(BaseModel):
    response: str
    elapsed_seconds: float
    model: str


class IngestRequest(BaseModel):
    docx_path: str = Field(..., description="DOCX 文件路径")


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


_start_time = time.time()


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """普通对话（一次性返回完整结果）"""
    agent = get_agent()
    start = time.time()

    try:
        response = agent.analyze(req.message)
    except Exception as e:
        logger.error(f"[API] 分析失败: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {e}")

    elapsed = time.time() - start
    return ChatResponse(
        response=response,
        elapsed_seconds=round(elapsed, 2),
        model=agent.llm.model,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式输出"""

    async def event_generator():
        agent = get_agent()
        start = time.time()

        try:
            for chunk in agent.analyze_stream(req.message):
                event_data = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

            elapsed = time.time() - start
            done_data = json.dumps({
                "done": True,
                "elapsed_seconds": round(elapsed, 2),
                "model": agent.llm.model,
            }, ensure_ascii=False)
            yield f"data: {done_data}\n\n"

        except Exception as e:
            logger.error(f"[SSE] 流式生成失败: {type(e).__name__}: {e}")
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    """增量录入法条"""
    if not os.path.exists(req.docx_path):
        raise HTTPException(status_code=400, detail=f"文件不存在: {req.docx_path}")

    result = ingest_law(req.docx_path)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "未知错误"))

    return result


@app.get("/api/pipeline/status")
async def pipeline_status():
    """查看管道检查点状态"""
    status = PipelineRunner.get_checkpoint_status()
    if status is None:
        return {"has_checkpoint": False}
    return {"has_checkpoint": True, "checkpoint": status}


def main():
    import uvicorn
    logger.info("启动 FastAPI 服务...")
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
