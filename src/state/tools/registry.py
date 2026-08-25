"""
src.state.tools.registry —— 工具注册中心（单例模式）

💡 面试讲解点：
    · 作用：Agent 不需要知道有多少个工具 / 每个工具叫什么名字 ——
            只需要从 Registry 按名字取，或者把可用工具列表传给 LLM 做 function calling
    · 为什么用单例：整个进程共享一份 Tool 实例（避免赔偿金计算器的社平工资数据重复装载）
    · 扩展：
        未来加新工具（如赔偿金计算器 doc_generator）：
            1) 写 Tool 子类
            2) 调 registry.register(MyTool())
            3) 完事儿 — 不需要改 Orchestrator 任何代码
        → 符合开闭原则
"""
from __future__ import annotations
from typing import Optional
from .base import BaseTool, ToolResult


class ToolRegistry:
    """
    工具注册中心（单例）：
        register(tool) → 注册
        get(name)      → 按名取
        run(name, **params) → 直接执行并返回 ToolResult
        list_all()     → 返回所有工具元信息（给 LLM / 调试面板）
        to_function_schemas() → 返回所有工具的 OpenAI 风格 function schema
    """

    _instance: Optional["ToolRegistry"] = None   # 单例实例

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: dict[str, BaseTool] = {}
        return cls._instance

    # ============== 管理 API ==============
    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            # 允许重新注册（V2 升级 Mock 工具用），但打个提醒
            import warnings
            warnings.warn(f"ToolRegistry: 覆盖已注册工具 {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def list_all(self) -> list[dict]:
        """给调试/系统状态面板看的简洁列表"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "required_params": list(t.input_schema.get("required", [])),
            }
            for t in self._tools.values()
        ]

    def to_function_schemas(self) -> list[dict]:
        """给 LLM function calling 用的标准 JSON Schema 列表"""
        return [t.to_function_schema() for t in self._tools.values()]

    def summarize_for_llm_text(self) -> str:
        """非 function-calling 模型：把可用工具拼一段文本描述加进 System Prompt"""
        lines = ["【可用工具列表】"]
        for t in self._tools.values():
            props = t.input_schema.get("properties", {})
            params_desc = "; ".join(
                f"{k}={v.get('type','any')}({v.get('description','')})"
                for k, v in props.items()
            )
            lines.append(f"  - {t.name}: {t.description}  参数: {params_desc}")
        lines.append("调用方式：[[TOOL:工具名|JSON参数]]，返回结果我会提供给你。")
        return "\n".join(lines)

    # ============== 直接执行 ==============
    def run(self, tool_name: str, **params) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name, success=False,
                error=f"工具不存在: {tool_name}。可用: {list(self._tools)}",
            )
        return tool.run(**params)


# 便捷导入：`from src.state.tools import get_registry`
def get_registry() -> ToolRegistry:
    return ToolRegistry()