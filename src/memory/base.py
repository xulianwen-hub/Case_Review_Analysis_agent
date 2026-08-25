"""
src.memory.base —— 所有记忆模块的抽象基类

💡 面试讲解点（为什么要抽象层？）
    · 遵循开闭原则：新增记忆类型（比如向量长期记忆）不需要改现有代码
    · 统一 API：Memory.read() / Memory.write() / Memory.clear()
      → 不管底层是 dict、SQLite 还是 Milvus，上层 Orchestrator 调用方式不变
    · 单测友好：可以写 InMemoryMock 快速测 Orchestrator 逻辑
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4


# ============================================================
# 通用数据结构：记忆项
# ============================================================
@dataclass
class MemoryItem:
    """一条独立的记忆条目"""
    role: str                    # user / assistant / system / rag_result / tool_result / slot_update
    content: Any                 # 具体内容：str / dict / RetrievedDoc list
    timestamp: float = 0.0       # 单调时间戳，排序用
    metadata: dict = field(default_factory=dict)
    item_id: str = field(default_factory=lambda: uuid4().hex[:12])
    tokens_estimate: int = 0     # token 数预估值（SummaryBuffer 用）


class BaseMemory(ABC):
    """所有 Memory 实现共同遵循的最小接口"""

    # ============== 子类必须实现 ==============
    @abstractmethod
    def write(self, role: str, content: Any, **metadata) -> MemoryItem:
        """写入一条记忆，返回创建好的 MemoryItem"""

    @abstractmethod
    def read(self, limit: Optional[int] = None, **filters) -> list[MemoryItem]:
        """
        读取记忆条目（按时间从旧到新排序）
        :param limit: 最多返回多少条（None=不限制）
        :param filters: 任意过滤条件，例如 role="user" 或 metadata["type"] = "slot"
        """

    @abstractmethod
    def clear(self) -> None:
        """清空该记忆实例的所有内容"""

    # ============== 子类可选覆盖 ==============
    def peek_latest(self, role: Optional[str] = None) -> Optional[MemoryItem]:
        """快速查看最新一条（可选按 role 过滤）"""
        items = self.read(limit=1)
        if not items:
            return None
        if role is None:
            return items[-1]
        for it in reversed(self.read()):
            if it.role == role:
                return it
        return None

    def total_items(self) -> int:
        return len(self.read())

    def total_tokens(self) -> int:
        return sum(it.tokens_estimate for it in self.read())