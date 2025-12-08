"""Base protocol for vision-language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from src.core.types import AgentEvent, Frame, Message, ToolDefinition


@dataclass
class ModelInfo:
    """Information about a model."""

    model_id: str
    provider: str
    display_name: str
    max_images: int
    supports_video: bool
    supports_tools: bool
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    context_window: int  # tokens


class VisionLanguageModel(ABC):
    """Abstract base class for vision-language models.

    All model providers (OpenAI, Anthropic, Google, etc.) implement this interface.

    The key method is `analyze()`, which:
    - Takes frames, optional audio transcript, tools, and context
    - Returns an async iterator of events (messages, tool calls)
    - Supports streaming for real-time responses
    """

    provider: str
    model_id: str

    @abstractmethod
    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames and audio, yielding events.

        Args:
            frames: List of video frames to analyze
            audio_transcript: Optional transcription of audio
            tools: List of available tools
            context: Previous conversation messages
            system_prompt: System instructions for the model

        Yields:
            Message or ToolCall events as they're generated
        """
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this model supports streaming responses."""
        ...

    @property
    @abstractmethod
    def max_images(self) -> int:
        """Maximum number of images per request."""
        ...

    @property
    @abstractmethod
    def cost_per_1k_input_tokens(self) -> float:
        """Cost in USD per 1000 input tokens."""
        ...

    @property
    @abstractmethod
    def cost_per_1k_output_tokens(self) -> float:
        """Cost in USD per 1000 output tokens."""
        ...

    @classmethod
    @abstractmethod
    def available_models(cls) -> list[ModelInfo]:
        """List all available models for this provider."""
        ...

    def _build_user_content(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
    ) -> str:
        """Build the user message content.

        Subclasses may override for provider-specific formatting.
        """
        parts = []

        if frames:
            parts.append(f"[{len(frames)} image(s) attached]")

        if audio_transcript:
            parts.append(f"Audio transcript: {audio_transcript}")

        if not parts:
            parts.append("Analyze the provided content.")

        return " ".join(parts)
