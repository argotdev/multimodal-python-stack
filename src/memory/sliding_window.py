"""Sliding window memory for context management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from src.core.types import AgentEvent, Message, ToolCall, ToolResult


@dataclass
class SlidingWindowMemory:
    """Simple sliding window context manager.

    Maintains a fixed-size window of recent messages,
    automatically removing old messages when the limit is reached.

    This is the recommended memory implementation for most use cases:
    - Simple and predictable behavior
    - Low overhead
    - Works well with all model providers

    Example:
        memory = SlidingWindowMemory(max_messages=20, max_tokens=4000)

        # Add events
        memory.add(Message(role="user", content="Hello"))
        memory.add(Message(role="assistant", content="Hi there!"))

        # Get context for model
        context = memory.get_context(max_tokens=2000)
    """

    max_messages: int = 20
    max_tokens: int = 4000

    # Internal state
    _messages: deque[Message] = field(default_factory=lambda: deque(maxlen=20), init=False)

    def __post_init__(self):
        # Recreate deque with correct maxlen
        self._messages = deque(maxlen=self.max_messages)

    def add(self, event: AgentEvent) -> None:
        """Add an event to memory.

        Converts tool calls and results to messages for context.

        Args:
            event: Message, ToolCall, or ToolResult to add
        """
        if isinstance(event, Message):
            self._messages.append(event)

        elif isinstance(event, ToolCall):
            # Store tool calls as assistant messages
            self._messages.append(
                Message(
                    role="assistant",
                    content=f"[Calling tool: {event.name}]",
                    timestamp=datetime.now(),
                    metadata={"tool_call": True, "tool_name": event.name},
                )
            )

        elif isinstance(event, ToolResult):
            # Store tool results as tool messages
            content = f"[Tool result: {event.output}]" if event.success else f"[Tool error: {event.error}]"
            self._messages.append(
                Message(
                    role="tool",
                    content=content,
                    timestamp=datetime.now(),
                    metadata={"tool_result": True, "success": event.success},
                )
            )

    def get_context(self, max_tokens: int | None = None) -> list[Message]:
        """Get recent context up to max_tokens.

        Returns messages from most recent to oldest, stopping when
        the token limit is reached.

        Args:
            max_tokens: Maximum tokens to include (uses default if not specified)

        Returns:
            List of messages for model context
        """
        limit = max_tokens or self.max_tokens
        result = []
        token_count = 0

        # Iterate from most recent to oldest
        for msg in reversed(self._messages):
            msg_tokens = self._estimate_tokens(msg.content)

            if token_count + msg_tokens > limit:
                break

            result.insert(0, msg)  # Insert at beginning to maintain order
            token_count += msg_tokens

        return result

    def get_all(self) -> list[Message]:
        """Get all messages in memory."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear all memory."""
        self._messages.clear()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token.
        This is a rough estimate but works well enough for context management.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4 + 1

    @property
    def message_count(self) -> int:
        """Number of messages in memory."""
        return len(self._messages)

    @property
    def estimated_tokens(self) -> int:
        """Estimated total tokens in memory."""
        return sum(self._estimate_tokens(m.content) for m in self._messages)

    def summarize(self) -> str:
        """Get a summary of memory contents.

        Useful for debugging and logging.

        Returns:
            Human-readable summary string
        """
        if not self._messages:
            return "Memory is empty"

        roles = {}
        for msg in self._messages:
            roles[msg.role] = roles.get(msg.role, 0) + 1

        role_summary = ", ".join(f"{k}: {v}" for k, v in roles.items())
        return (
            f"Memory: {self.message_count} messages, "
            f"~{self.estimated_tokens} tokens ({role_summary})"
        )


@dataclass
class ConversationMemory(SlidingWindowMemory):
    """Extended memory with conversation history tracking.

    Adds features for managing multi-turn conversations:
    - Session tracking
    - Message searching
    - Export/import
    """

    session_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.session_id:
            import uuid
            self.session_id = str(uuid.uuid4())[:8]

    def find_messages(self, content_contains: str) -> list[Message]:
        """Find messages containing a string.

        Args:
            content_contains: String to search for

        Returns:
            List of matching messages
        """
        return [
            msg for msg in self._messages
            if content_contains.lower() in msg.content.lower()
        ]

    def get_by_role(self, role: str) -> list[Message]:
        """Get all messages with a specific role.

        Args:
            role: Role to filter by (user, assistant, system, tool)

        Returns:
            List of messages with that role
        """
        return [msg for msg in self._messages if msg.role == role]

    def export(self) -> list[dict]:
        """Export memory to serializable format.

        Returns:
            List of message dictionaries
        """
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata,
            }
            for msg in self._messages
        ]

    def import_messages(self, messages: list[dict]) -> None:
        """Import messages from serialized format.

        Args:
            messages: List of message dictionaries
        """
        for msg_data in messages:
            timestamp = datetime.fromisoformat(msg_data.get("timestamp", datetime.now().isoformat()))
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=timestamp,
                metadata=msg_data.get("metadata", {}),
            )
            self._messages.append(msg)
