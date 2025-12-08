"""Fireworks.ai vision model implementation."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import AsyncIterator

import httpx

from src.core.types import AgentEvent, Frame, Message, ToolCall, ToolDefinition
from src.models.base import ModelInfo, VisionLanguageModel


class FireworksVisionModel(VisionLanguageModel):
    """Fireworks.ai vision models.

    Features:
    - Good price/performance ratio
    - Open models (LLaVA, etc.)
    - Fast inference
    - Tool calling support

    Example:
        model = FireworksVisionModel(model_id="firellava-13b")
        async for event in model.analyze(frames, ...):
            print(event)
    """

    provider = "fireworks"

    MODELS = {
        "firellava-13b": ModelInfo(
            model_id="accounts/fireworks/models/firellava-13b",
            provider="fireworks",
            display_name="FireLLaVA 13B",
            max_images=4,
            supports_video=False,
            supports_tools=False,  # Limited tool support
            cost_per_1k_input=0.0002,
            cost_per_1k_output=0.0002,
            context_window=4096,
        ),
        "phi-3-vision-128k-instruct": ModelInfo(
            model_id="accounts/fireworks/models/phi-3-vision-128k-instruct",
            provider="fireworks",
            display_name="Phi-3 Vision 128K",
            max_images=4,
            supports_video=False,
            supports_tools=False,
            cost_per_1k_input=0.0002,
            cost_per_1k_output=0.0002,
            context_window=128000,
        ),
    }

    # Map short names to full model IDs
    MODEL_ID_MAP = {
        "firellava-13b": "accounts/fireworks/models/firellava-13b",
        "phi-3-vision-128k-instruct": "accounts/fireworks/models/phi-3-vision-128k-instruct",
    }

    BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(
        self,
        model_id: str = "firellava-13b",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """Initialize Fireworks model.

        Args:
            model_id: Model to use (short name or full ID)
            api_key: Fireworks API key (uses env var if not provided)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
        """
        # Handle short names
        if model_id in self.MODEL_ID_MAP:
            self._full_model_id = self.MODEL_ID_MAP[model_id]
            self.model_id = model_id
        else:
            self._full_model_id = model_id
            self.model_id = model_id

        self.max_tokens = max_tokens
        self.temperature = temperature

        if model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(self.MODELS.keys())}")

        self._info = self.MODELS[model_id]
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")

    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames using Fireworks."""
        messages = self._build_messages(frames, audio_transcript, context, system_prompt)

        # Note: Fireworks vision models have limited tool support
        # We'll include tools in the prompt if provided
        if tools:
            tool_descriptions = self._tools_to_prompt(tools)
            if messages and messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                if isinstance(content, list):
                    # Find text part and append
                    for part in content:
                        if part.get("type") == "text":
                            part["text"] += f"\n\nAvailable tools:\n{tool_descriptions}"
                            break

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._full_model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                current_content = ""
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})

                        if delta.get("content"):
                            content = delta["content"]
                            current_content += content
                            yield Message(
                                role="assistant",
                                content=content,
                                timestamp=datetime.now(),
                                metadata={"chunk": True},
                            )
                    except json.JSONDecodeError:
                        continue

        # Try to parse tool calls from the response
        if tools and current_content:
            tool_calls = self._parse_tool_calls(current_content, tools)
            for tc in tool_calls:
                yield tc

    def _build_messages(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        context: list[Message],
        system_prompt: str,
    ) -> list[dict]:
        """Build Fireworks message format (OpenAI compatible)."""
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

    def _tools_to_prompt(self, tools: list[ToolDefinition]) -> str:
        """Convert tools to a prompt description."""
        descriptions = []
        for t in tools:
            params = json.dumps(t.parameters, indent=2)
            descriptions.append(f"- {t.name}: {t.description}\n  Parameters: {params}")
        return "\n".join(descriptions)

    def _parse_tool_calls(
        self, content: str, tools: list[ToolDefinition]
    ) -> list[ToolCall]:
        """Try to parse tool calls from response content."""
        # Simple heuristic: look for JSON-like structures
        tool_calls = []

        for tool in tools:
            if tool.name.lower() in content.lower():
                # Try to extract arguments
                try:
                    # Look for JSON block
                    import re
                    json_match = re.search(r'\{[^{}]*\}', content)
                    if json_match:
                        args = json.loads(json_match.group())
                        tool_calls.append(ToolCall(name=tool.name, arguments=args))
                except (json.JSONDecodeError, AttributeError):
                    pass

        return tool_calls

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
