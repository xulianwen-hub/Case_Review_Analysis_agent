"""
src.memory —— 记忆 & 上下文管理模块（秋招面试核心模块）

子模块 & 面试要点：
    base            MemoryItem / BaseMemory 抽象层
                    → 开闭原则 / 多态统一接口

    short_term      ShortTermMemory：滑动窗口短期记忆 + RAG/Tool 结果缓存
                    → 为什么用滑动窗口 + 去重裁剪

    evidence_slots  EvidenceSlotMemory：证据槽（P0/P1 优先级 + 三态）
                    → 领域 Agent vs 通用 Chatbot：结构化槽驱动追问

    summary_buffer  SummaryBuffer：滚动渐进式对话压缩
                    → Context Window 爆炸的经典解决方案（滑动窗口+摘要的权衡）
"""
from .base import BaseMemory, MemoryItem  # noqa: F401
from .short_term import ShortTermMemory  # noqa: F401
from .evidence_slots import (  # noqa: F401
    SlotState, SlotPriority, EvidenceSlot, EvidenceSlotMemory,
)
from .summary_buffer import SummaryBuffer  # noqa: F401

__all__ = [
    "BaseMemory", "MemoryItem",
    "ShortTermMemory",
    "SlotState", "SlotPriority", "EvidenceSlot", "EvidenceSlotMemory",
    "SummaryBuffer",
]