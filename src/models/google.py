"""Google Gemini vision model implementation."""

from __future__ import annotations

import os
from datetime import datetime
from typing import AsyncIterator

import google.generativeai as genai
from PIL import Image
import io

from src.core.types import AgentEvent, Frame, Message, ToolCall, ToolDefinition
from src.models.base import ModelInfo, VisionLanguageModel


class GoogleVisionModel(VisionLanguageModel):
    """Google Gemini 1.5 Flash and Pro vision models.

    Features:
    - Extremely low cost
    - Massive context window (up to 1M tokens)
    - Native video understanding
    - Up to 3600 images per request

    Example:
        model = GoogleVisionModel(model_id="gemini-1.5-flash")
        async for event in model.analyze(frames, ...):
            print(event)
    """

    provider = "google"

    MODELS = {
        # Gemini 3 Family (November 2025 - Latest)
        "gemini-3.0-pro": ModelInfo(
            model_id="gemini-3.0-pro",
            provider="google",
            display_name="Gemini 3.0 Pro",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
            context_window=2000000,
        ),
        "gemini-3.0-flash": ModelInfo(
            model_id="gemini-3.0-flash",
            provider="google",
            display_name="Gemini 3.0 Flash",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            context_window=1000000,
        ),
        # Gemini 2.5 Family
        "gemini-2.5-pro": ModelInfo(
            model_id="gemini-2.5-pro-preview-06-05",
            provider="google",
            display_name="Gemini 2.5 Pro",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.01,
            context_window=1000000,
        ),
        "gemini-2.5-flash": ModelInfo(
            model_id="gemini-2.5-flash-preview-05-20",
            provider="google",
            display_name="Gemini 2.5 Flash",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            context_window=1000000,
        ),
        "gemini-2.5-flash-lite": ModelInfo(
            model_id="gemini-2.5-flash-lite",
            provider="google",
            display_name="Gemini 2.5 Flash Lite",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.000075,
            cost_per_1k_output=0.0003,
            context_window=1000000,
        ),
        # Gemini 2.0 Family
        "gemini-2.0-flash": ModelInfo(
            model_id="gemini-2.0-flash",
            provider="google",
            display_name="Gemini 2.0 Flash",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            context_window=1000000,
        ),
        # Gemini 1.5 Family (Legacy)
        "gemini-1.5-flash": ModelInfo(
            model_id="gemini-1.5-flash",
            provider="google",
            display_name="Gemini 1.5 Flash",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.000075,
            cost_per_1k_output=0.0003,
            context_window=1000000,
        ),
        "gemini-1.5-pro": ModelInfo(
            model_id="gemini-1.5-pro",
            provider="google",
            display_name="Gemini 1.5 Pro",
            max_images=3600,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
            context_window=2000000,
        ),
    }

    def __init__(
        self,
        model_id: str = "gemini-1.5-flash",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """Initialize Google model.

        Args:
            model_id: Model to use
            api_key: Google API key (uses env var if not provided)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        if model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(self.MODELS.keys())}")

        self._info = self.MODELS[model_id]

        # Configure the API
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))

        # Create model
        self.model = genai.GenerativeModel(model_id)

    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames using Gemini."""
        # Build content parts
        content_parts = []

        # Add system prompt as first text part
        if system_prompt:
            content_parts.append(f"System: {system_prompt}\n\n")

        # Add context as text
        for msg in context:
            content_parts.append(f"{msg.role.capitalize()}: {msg.content}\n")

        # Add images (convert Frame to PIL Image)
        for frame in frames[: self.max_images]:
            pil_image = Image.fromarray(frame.data)
            content_parts.append(pil_image)

        # Add user text prompt
        text = self._build_user_content(frames, audio_transcript)
        content_parts.append(f"\nUser: {text}")

        # Configure tools if provided
        tool_config = None
        if tools:
            # Convert tools to Gemini format
            function_declarations = []
            for t in tools:
                function_declarations.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            tool_config = genai.types.Tool(function_declarations=function_declarations)

        # Generate content (Gemini doesn't have true streaming for vision)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response = await self.model.generate_content_async(
            content_parts,
            generation_config=generation_config,
            tools=[tool_config] if tool_config else None,
        )

        # Process response
        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                # Handle text
                if hasattr(part, "text") and part.text:
                    yield Message(
                        role="assistant",
                        content=part.text,
                        timestamp=datetime.now(),
                    )

                # Handle function calls
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    yield ToolCall(
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    )

    @property
    def supports_streaming(self) -> bool:
        # Gemini has limited streaming support for vision
        return False

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
