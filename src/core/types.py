"""Core type definitions for multimodal agents."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
from PIL import Image


@dataclass
class Frame:
    """A single video frame."""

    data: np.ndarray  # RGB image array (H, W, 3)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"

    def to_base64(self, format: str = "JPEG", quality: int = 85) -> str:
        """Convert frame to base64-encoded string for API calls."""
        img = Image.fromarray(self.data)
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def resize(self, max_size: int = 512) -> Frame:
        """Resize frame maintaining aspect ratio."""
        img = Image.fromarray(self.data)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return Frame(
            data=np.array(img),
            timestamp=self.timestamp,
            source=self.source,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (height, width, channels)."""
        return self.data.shape


@dataclass
class AudioChunk:
    """A chunk of audio data."""

    data: np.ndarray  # Audio samples
    sample_rate: int = 16000
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"

    @property
    def duration_seconds(self) -> float:
        """Duration of this chunk in seconds."""
        return len(self.data) / self.sample_rate


@dataclass
class Message:
    """A message in the conversation context."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class ToolCall:
    """A tool call request from the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str = ""

    def __post_init__(self):
        if not self.call_id:
            import uuid
            self.call_id = str(uuid.uuid4())[:8]


@dataclass
class ToolResult:
    """Result of a tool execution."""

    output: Any = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the tool executed successfully."""
        return self.error is None


@dataclass
class ToolDefinition:
    """Definition of a tool for model APIs."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_google_format(self) -> dict[str, Any]:
        """Convert to Google/Gemini tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# Type alias for all events the agent can produce
AgentEvent = Message | ToolCall | ToolResult
