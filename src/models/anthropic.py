"""Anthropic Claude vision model implementation."""

from __future__ import annotations

import os
from datetime import datetime
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from src.core.types import AgentEvent, Frame, Message, ToolCall, ToolDefinition
from src.models.base import ModelInfo, VisionLanguageModel


class AnthropicVisionModel(VisionLanguageModel):
    """Anthropic Claude 3.5 Sonnet and Haiku vision models.

    Features:
    - Excellent reasoning capabilities
    - Up to 20 images per request
    - Strong tool calling
    - Streaming support

    Example:
        model = AnthropicVisionModel(model_id="claude-3-5-haiku-latest")
        async for event in model.analyze(frames, ...):
            print(event)
    """

    provider = "anthropic"

    MODELS = {
        # Claude 4.5 Family (November 2025 - Latest)
        "claude-opus-4-5-20251101": ModelInfo(
            model_id="claude-opus-4-5-20251101",
            provider="anthropic",
            display_name="Claude Opus 4.5",
            max_images=20,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.025,
            context_window=200000,
        ),
        "claude-sonnet-4-5": ModelInfo(
            model_id="claude-sonnet-4-5",
            provider="anthropic",
            display_name="Claude Sonnet 4.5",
            max_images=20,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            context_window=200000,
        ),
        "claude-haiku-4-5": ModelInfo(
            model_id="claude-haiku-4-5",
            provider="anthropic",
            display_name="Claude Haiku 4.5",
            max_images=20,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.005,
            context_window=200000,
        ),
        # Claude 4 Family (May 2025)
        "claude-opus-4-20250514": ModelInfo(
            model_id="claude-opus-4-20250514",
            provider="anthropic",
            display_name="Claude Opus 4",
            max_images=20,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            context_window=200000,
        ),
        "claude-sonnet-4-20250514": ModelInfo(
            model_id="claude-sonnet-4-20250514",
            provider="anthropic",
            display_name="Claude Sonnet 4",
            max_images=20,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            context_window=200000,
        ),
        # Claude 3.5 Family (Legacy)
        "claude-3-5-sonnet-latest": ModelInfo(
            model_id="claude-3-5-sonnet-latest",
            provider="anthropic",
            display_name="Claude 3.5 Sonnet",
            max_images=20,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            context_window=200000,
        ),
        "claude-3-5-haiku-latest": ModelInfo(
            model_id="claude-3-5-haiku-latest",
            provider="anthropic",
            display_name="Claude 3.5 Haiku",
            max_images=20,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
            context_window=200000,
        ),
    }

    def __init__(
        self,
        model_id: str = "claude-3-5-haiku-latest",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """Initialize Anthropic model.

        Args:
            model_id: Model to use
            api_key: Anthropic API key (uses env var if not provided)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        if model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(self.MODELS.keys())}")

        self._info = self.MODELS[model_id]
        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames using Claude."""
        messages = self._build_messages(frames, audio_transcript, context)

        # Convert tools to Anthropic format
        anthropic_tools = [t.to_anthropic_format() for t in tools] if tools else None

        # Create streaming message
        async with self.client.messages.stream(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system_prompt if system_prompt else None,
            messages=messages,
            tools=anthropic_tools,
            temperature=self.temperature,
        ) as stream:
            current_tool_use = None

            async for event in stream:
                # Handle text content
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield Message(
                            role="assistant",
                            content=event.delta.text,
                            timestamp=datetime.now(),
                            metadata={"chunk": True},
                        )
                    elif hasattr(event.delta, "partial_json"):
                        # Tool use JSON delta
                        if current_tool_use:
                            current_tool_use["partial_json"] += event.delta.partial_json

                # Handle tool use start
                elif event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            current_tool_use = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "partial_json": "",
                            }

                # Handle tool use completion
                elif event.type == "content_block_stop":
                    if current_tool_use:
                        try:
                            import json
                            args = json.loads(current_tool_use["partial_json"]) if current_tool_use["partial_json"] else {}
                        except json.JSONDecodeError:
                            args = {}

                        yield ToolCall(
                            name=current_tool_use["name"],
                            arguments=args,
                            call_id=current_tool_use["id"],
                        )
                        current_tool_use = None

    def _build_messages(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        context: list[Message],
    ) -> list[dict]:
        """Build Anthropic message format with images."""
        messages = []

        # Context messages
        for msg in context:
            messages.append(msg.to_dict())

        # Current user message with images
        content = []

        # Add frames as images (limit to max_images)
        for frame in frames[: self.max_images]:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame.to_base64(),
                },
            })

        # Add text prompt
        text = self._build_user_content(frames, audio_transcript)
        content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content})

        return messages

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def max_images(self) -> int:
        return self._info.max_images

    @property
    def cost_per_1k_input_tokens(self) -> float:
        return self._info.cost_per_1k_input

    @property
    def cost_per_1k_output_tokens(self) -> float:
        return self._info.cost_per_1k_output

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        return list(cls.MODELS.values())
