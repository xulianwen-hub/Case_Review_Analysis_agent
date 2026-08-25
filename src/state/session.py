"""
src.state.session —— Session 管理器（把 FSM + Memory 绑定到 session_id，支持多用户并发）

💡 面试讲解点：
    · 为什么需要 Session？—— Agent 不是单轮函数，用户会连问 5-10 轮：
        ① session_id 把「同一个人/同一个案子」的所有消息绑定在一起
        ② 让 Agent 在轮与轮之间记住 FSM 状态、证据槽、对话历史等全部上下文
    · Session 内部装什么：
        session_id + created_at + FSM + 4 个 Memory（短期/证据槽/摘要/长期） + 用户元信息
    · SessionStore：
        V1 用内存 dict（够开发调试 + 面试讲）
        V2 只需要把 store 换成 SQLite/Redis：接口保持 get/put/delete 不变
        → 这就是「面向接口存储」的典型面试亮点：替换持久化方案不需要改 Session 本身
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from .fsm import DialogueFSM, AgentState
from ..memory import ShortTermMemory, EvidenceSlotMemory, SummaryBuffer, BaseMemory


# ============================================================
# Session 对象：单个用户的一次咨询全量上下文
# ============================================================
@dataclass
class AgentSession:
    session_id: str
    fsm: DialogueFSM
    short_term: ShortTermMemory
    evidence_slots: EvidenceSlotMemory
    summary: SummaryBuffer
    created_at: float
    last_access_at: float = 0.0
    user_meta: dict = field(default_factory=dict)  # user_id / city / 来源（微信/网页/API）
    ttl_seconds: int = 3600                        # 1 小时无访问自动清理

    def touch(self) -> None:
        self.last_access_at = time.time()

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        return (now - self.last_access_at) > self.ttl_seconds

    def reset_all(self) -> None:
        """用户点「重新开始」时一键重置所有状态但保留 session_id"""
        self.fsm.reset()
        self.short_term.clear()
        self.evidence_slots.clear()
        self.summary.clear()


# ============================================================
# SessionStore：按 session_id 存取（V1 = 内存，可替换）
# ============================================================
class SessionStore:
    """
    Session 存储抽象（V1 内存实现）。

    未来扩展到 Redis/SQLite：只需写同名类实现 get()/put()/delete()/cleanup_expired()
    四个方法，Orchestrator 代码不需要改 —— 里氏替换原则。
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        self._sessions: dict[str, AgentSession] = {}
        self.default_ttl = default_ttl_seconds

    # ============== CRUD ==============
    def create(self, session_id: Optional[str] = None,
               **user_meta) -> AgentSession:
        """创建新 Session（附带 FSM + 3 种 Memory 的依赖注入）"""
        sid = session_id or uuid4().hex[:16]
        now = time.time()
        sess = AgentSession(
            session_id=sid,
            fsm=DialogueFSM(initial_state=AgentState.INIT),
            short_term=ShortTermMemory(max_dialogue_pairs=30),
            evidence_slots=EvidenceSlotMemory(),
            summary=SummaryBuffer(token_threshold=4000, message_threshold=20),
            created_at=now, last_access_at=now,
            user_meta=user_meta, ttl_seconds=self.default_ttl,
        )
        self._sessions[sid] = sess
        return sess

    def get(self, session_id: str) -> Optional[AgentSession]:
        sess = self._sessions.get(session_id)
        if sess is None:
            return None
        if sess.is_expired():
            self.delete(session_id)
            return None
        sess.touch()
        return sess

    def get_or_create(self, session_id: Optional[str],
                      **user_meta) -> tuple[AgentSession, bool]:
        """返回 (session, is_created_now)：Orchestrator 每次处理消息都用这个"""
        if session_id:
            sess = self.get(session_id)
            if sess is not None:
                return sess, False
        return self.create(session_id, **user_meta), True

    def put(self, session: AgentSession) -> None:
        session.touch()
        self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def total_sessions(self) -> int:
        return len(self._sessions)

    # ============== 过期清理（Orchestrator 后台线程可周期性调用） ==============
    def cleanup_expired(self) -> int:
        now = time.time()
        expired_ids = [sid for sid, s in self._sessions.items() if s.is_expired(now)]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)


# ============================================================
# 便捷单例（和 ToolRegistry 保持一致的使用方式）
# ============================================================
_GLOBAL_STORE: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = SessionStore()
    return _GLOBAL_STORE