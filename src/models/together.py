"""Together.ai vision model implementation."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import AsyncIterator

from together import AsyncTogether

from src.core.types import AgentEvent, Frame, Message, ToolCall, ToolDefinition
from src.models.base import ModelInfo, VisionLanguageModel


class TogetherVisionModel(VisionLanguageModel):
    """Together.ai vision models.

    Features:
    - Open source models
    - Good price/performance
    - Llama 3.2 Vision support
    - Tool calling support

    Example:
        model = TogetherVisionModel(model_id="llama-3.2-11b-vision")
        async for event in model.analyze(frames, ...):
            print(event)
    """

    provider = "together"

    MODELS = {
        "llama-3.2-11b-vision": ModelInfo(
            model_id="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            provider="together",
            display_name="Llama 3.2 11B Vision",
            max_images=4,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.00018,
            cost_per_1k_output=0.00018,
            context_window=131072,
        ),
        "llama-3.2-90b-vision": ModelInfo(
            model_id="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            provider="together",
            display_name="Llama 3.2 90B Vision",
            max_images=4,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.0012,
            cost_per_1k_output=0.0012,
            context_window=131072,
        ),
    }

    # Map short names to full model IDs
    MODEL_ID_MAP = {
        "llama-3.2-11b-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "llama-3.2-90b-vision": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    }

    def __init__(
        self,
        model_id: str = "llama-3.2-11b-vision",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """Initialize Together model.

        Args:
            model_id: Model to use (short name or full ID)
            api_key: Together API key (uses env var if not provided)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
        """
        # Handle short names
        if model_id in self.MODEL_ID_MAP:
            self._full_model_id = self.MODEL_ID_MAP[model_id]
            self.model_id = model_id
        else:
            self._full_model_id = model_id
            # Find short name
            for short, full in self.MODEL_ID_MAP.items():
                if full == model_id:
                    self.model_id = short
                    break
            else:
                self.model_id = model_id

        self.max_tokens = max_tokens
        self.temperature = temperature

        if self.model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(self.MODELS.keys())}")

        self._info = self.MODELS[self.model_id]
        self.client = AsyncTogether(api_key=api_key or os.getenv("TOGETHER_API_KEY"))

    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames using Together."""
        messages = self._build_messages(frames, audio_transcript, context, system_prompt)

        # Convert tools to OpenAI-compatible format
        together_tools = [t.to_openai_format() for t in tools] if tools else None

        # Create streaming completion
        stream = await self.client.chat.completions.create(
            model=self._full_model_id,
            messages=messages,
            tools=together_tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        # Track accumulated content and tool calls
        current_content = ""
        tool_calls_data: dict[int, dict] = {}

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
            if hasattr(delta, "tool_calls") and delta.tool_calls:
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
        """Build Together message format (OpenAI compatible)."""
        messages = []

        # System message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Context messages
        for msg in context:
            messages.append(msg.to_dict())

        # Current user message with images
        content = []

        # Add frames as images
        for frame in frames[: self.max_images]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame.to_base64()}",
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
