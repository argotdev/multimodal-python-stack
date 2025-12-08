"""OpenAI vision model implementation (GPT-4o, GPT-4o-mini)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.core.types import AgentEvent, Frame, Message, ToolCall, ToolDefinition
from src.models.base import ModelInfo, VisionLanguageModel


class OpenAIVisionModel(VisionLanguageModel):
    """OpenAI GPT-4o and GPT-4o-mini vision models.

    Features:
    - Excellent tool calling
    - Up to 10 images per request
    - Streaming support
    - Reliable and fast

    Example:
        model = OpenAIVisionModel(model_id="gpt-4o-mini")
        async for event in model.analyze(frames, ...):
            print(event)
    """

    provider = "openai"

    MODELS = {
        "gpt-4o": ModelInfo(
            model_id="gpt-4o",
            provider="openai",
            display_name="GPT-4o",
            max_images=10,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            context_window=128000,
        ),
        "gpt-4o-mini": ModelInfo(
            model_id="gpt-4o-mini",
            provider="openai",
            display_name="GPT-4o Mini",
            max_images=10,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            context_window=128000,
        ),
    }

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        image_detail: str = "low",
    ):
        """Initialize OpenAI model.

        Args:
            model_id: Model to use (gpt-4o or gpt-4o-mini)
            api_key: OpenAI API key (uses env var if not provided)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
            image_detail: Image detail level (low, high, auto)
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.image_detail = image_detail

        if model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(self.MODELS.keys())}")

        self._info = self.MODELS[model_id]
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames using GPT-4o."""
        messages = self._build_messages(frames, audio_transcript, context, system_prompt)

        # Convert tools to OpenAI format
        openai_tools = [t.to_openai_format() for t in tools] if tools else None

        # Create streaming completion
        stream = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=openai_tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        # Track accumulated content and tool calls
        current_content = ""
        tool_calls_data: dict[int, dict] = {}  # index -> {name, arguments}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Handle text content
            if delta.content:
                current_content += delta.content
                yield Message(
                    role="assistant",
                    content=delta.content,
                    timestamp=datetime.now(),
                    metadata={"chunk": True},
                )

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["arguments"] += tc.function.arguments

        # Yield complete tool calls after streaming
        for idx in sorted(tool_calls_data.keys()):
            tc_data = tool_calls_data[idx]
            if tc_data["name"]:
                try:
                    args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}

                yield ToolCall(
                    name=tc_data["name"],
                    arguments=args,
                    call_id=tc_data["id"],
                )

    def _build_messages(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        context: list[Message],
        system_prompt: str,
    ) -> list[dict]:
        """Build OpenAI message format with images."""
        messages = []

        # System message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Context messages
        for msg in context:
            messages.append(msg.to_dict())

        # Current user message with images
        content = []

        # Add frames as images (limit to max_images)
        for frame in frames[: self.max_images]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame.to_base64()}",
                    "detail": self.image_detail,
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
