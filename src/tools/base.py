"""Base tool protocol and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.types import ToolDefinition, ToolResult


class Tool(ABC):
    """Abstract base class for all tools.

    Tools are executable actions that the agent can take.
    Each tool has a name, description, and JSON Schema parameters.

    Example:
        class MyTool(Tool):
            name = "my_tool"
            description = "Does something useful"
            parameters = {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First argument"}
                },
                "required": ["arg1"]
            }

            async def execute(self, arg1: str) -> ToolResult:
                return ToolResult(output=f"Got {arg1}")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with output or error
        """
        ...

    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for model APIs."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolRegistry:
    """Registry for managing tools.

    Example:
        registry = ToolRegistry()
        registry.register(SlackAlertTool(...))
        registry.register(NotionTool(...))

        # Get tool by name
        tool = registry.get("send_slack_alert")

        # List all definitions
        definitions = registry.list_definitions()
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions for model APIs."""
        return [tool.to_definition() for tool in self._tools.values()]

    def list_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
