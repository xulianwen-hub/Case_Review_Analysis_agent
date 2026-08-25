"""
src.state.tools.base —— Tool 抽象基类（所有工具统一接口）

💡 面试讲解点：
    · 为什么要给 Tool 定义基类（而不是随便写函数）：
      ① 统一元数据：name/description/input_schema → 未来转 OpenAI function calling
        只需要 to_openai_schema() 一个方法就行（自动转 JSON Schema）
      ② 统一错误处理：超时、参数校验、异常捕获 → 不用每个工具写 try/except
      ③ 可发现性：ToolRegistry.summarize_available() 把所有工具列给 LLM 看
      ④ 可追踪：所有工具调用统一打 log / 存进 ShortTermMemory.tool_cache
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time


# ============================================================
# 工具调用结果（标准化，Orchestrator 统一处理）
# ============================================================
@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: int = 0          # 耗时，可观测性
    metadata: dict = field(default_factory=dict)

    def to_llm_text(self) -> str:
        """把结果格式化成 LLM 看得懂的自然语言片段（拼 Prompt 用）"""
        if not self.success:
            return f"[工具 {self.tool_name} 调用失败: {self.error}]"
        if isinstance(self.output, (dict, list)):
            import json
            try:
                return json.dumps(self.output, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return str(self.output)


class BaseTool(ABC):
    """
    所有自定义工具的基类。子类最少必须实现：
        name: str          唯一短名（英文下划线，如 compensation_calc）
        description: str   给 LLM/用户看的 1-2 句功能介绍
        input_schema: dict JSON Schema 风格的参数说明
        run(params) -> ToolResult
    """

    # ============== 子类必须覆盖 ==============
    name: str = "base_tool"
    description: str = ""
    input_schema: dict = field(default_factory=dict)  # {"properties": {...}, "required": [...]}

    @abstractmethod
    def _run(self, params: dict) -> Any:
        """真正的业务逻辑，抛异常也没关系，外层 run() 会捕获"""

    # ============== 通用框架层（子类一般不覆盖） ==============
    def __init__(self):
        if not self.description:
            self.description = f"{self.name} 工具"

    # ---- 参数校验（V1 轻量实现：检查 required 字段在不在） ----
    def validate_params(self, params: dict) -> Optional[str]:
        """返回 None 表示校验通过，否则返回错误描述"""
        required = self.input_schema.get("required", [])
        missing = [k for k in required if k not in params or params[k] is None]
        if missing:
            return f"缺少必填参数: {', '.join(missing)}"
        return None

    # ---- 统一包装入口：参数校验 + 计时 + 异常捕获 ----
    def run(self, **params) -> ToolResult:
        t0 = time.perf_counter()
        # 1. 参数校验
        err = self.validate_params(params)
        if err:
            return ToolResult(
                tool_name=self.name, success=False, error=err,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        # 2. 执行业务
        try:
            output = self._run(params)
            return ToolResult(
                tool_name=self.name, success=True, output=output,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"{type(e).__name__}: {str(e)}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )

    # ---- 自动转 OpenAI / DeepSeek function calling Schema ----
    def to_function_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.input_schema.get("properties", {}),
                    "required": self.input_schema.get("required", []),
                },
            },
        }