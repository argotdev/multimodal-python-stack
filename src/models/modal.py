"""Modal serverless vision model implementation.

Modal allows you to deploy any model serverlessly. This implementation
provides a client to call vision models hosted on Modal.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import AsyncIterator

import httpx

from src.core.types import AgentEvent, Frame, Message, ToolCall, ToolDefinition
from src.models.base import ModelInfo, VisionLanguageModel


class ModalVisionModel(VisionLanguageModel):
    """Modal serverless vision model client.

    Modal (modal.com) lets you run any model serverlessly with auto-scaling.
    This client calls a Modal endpoint that wraps a vision model.

    You need to deploy a Modal function that accepts:
    - images: list of base64-encoded images
    - prompt: the text prompt
    - system_prompt: optional system prompt
    - tools: optional tool definitions

    Example Modal function (deploy this on Modal):

        import modal

        app = modal.App("vision-model")

        @app.function(gpu="A10G", image=modal.Image.debian_slim().pip_install("transformers", "torch", "pillow"))
        @modal.web_endpoint(method="POST")
        def analyze(images: list[str], prompt: str, system_prompt: str = "", tools: list = None):
            # Your model inference code here
            # Return: {"text": "...", "tool_calls": [...]}
            pass

    Example usage:

        model = ModalVisionModel(
            endpoint_url="https://your-workspace--vision-model-analyze.modal.run",
            model_id="llama-3.2-90b-vision",
        )
        async for event in model.analyze(frames, ...):
            print(event)
    """

    provider = "modal"

    # Default model info - can be overridden per instance
    DEFAULT_INFO = ModelInfo(
        model_id="custom",
        provider="modal",
        display_name="Modal Custom Model",
        max_images=10,
        supports_video=False,
        supports_tools=True,
        cost_per_1k_input=0.0,  # Depends on your Modal setup
        cost_per_1k_output=0.0,
        context_window=128000,
    )

    # Pre-configured Modal model profiles
    MODELS = {
        "llama-3.2-90b-vision": ModelInfo(
            model_id="llama-3.2-90b-vision",
            provider="modal",
            display_name="Llama 3.2 90B Vision (Modal)",
            max_images=10,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.0009,  # Estimated based on A100 pricing
            cost_per_1k_output=0.0009,
            context_window=128000,
        ),
        "llama-3.2-11b-vision": ModelInfo(
            model_id="llama-3.2-11b-vision",
            provider="modal",
            display_name="Llama 3.2 11B Vision (Modal)",
            max_images=10,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.0002,  # Estimated based on A10G pricing
            cost_per_1k_output=0.0002,
            context_window=128000,
        ),
        "qwen2-vl-72b": ModelInfo(
            model_id="qwen2-vl-72b",
            provider="modal",
            display_name="Qwen2-VL 72B (Modal)",
            max_images=10,
            supports_video=True,
            supports_tools=True,
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.0008,
            context_window=32000,
        ),
        "pixtral-12b": ModelInfo(
            model_id="pixtral-12b",
            provider="modal",
            display_name="Pixtral 12B (Modal)",
            max_images=10,
            supports_video=False,
            supports_tools=False,
            cost_per_1k_input=0.0002,
            cost_per_1k_output=0.0002,
            context_window=128000,
        ),
    }

    def __init__(
        self,
        endpoint_url: str | None = None,
        model_id: str = "custom",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: float = 120.0,
        cost_per_1k_input: float | None = None,
        cost_per_1k_output: float | None = None,
    ):
        """Initialize Modal model client.

        Args:
            endpoint_url: Modal web endpoint URL (or set MODAL_ENDPOINT_URL env var)
            model_id: Model identifier (for tracking, can be any string)
            api_key: Optional API key for authenticated endpoints (or MODAL_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            cost_per_1k_input: Override cost tracking for input tokens
            cost_per_1k_output: Override cost tracking for output tokens
        """
        self.endpoint_url = endpoint_url or os.getenv("MODAL_ENDPOINT_URL")
        if not self.endpoint_url:
            raise ValueError(
                "endpoint_url required. Pass it directly or set MODAL_ENDPOINT_URL env var."
            )

        self.model_id = model_id
        self.api_key = api_key or os.getenv("MODAL_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Get model info
        if model_id in self.MODELS:
            self._info = self.MODELS[model_id]
        else:
            self._info = ModelInfo(
                model_id=model_id,
                provider="modal",
                display_name=f"Modal: {model_id}",
                max_images=10,
                supports_video=False,
                supports_tools=True,
                cost_per_1k_input=cost_per_1k_input or 0.0,
                cost_per_1k_output=cost_per_1k_output or 0.0,
                context_window=128000,
            )

        # Override costs if provided
        if cost_per_1k_input is not None:
            self._info = ModelInfo(
                **{**self._info.__dict__, "cost_per_1k_input": cost_per_1k_input}
            )
        if cost_per_1k_output is not None:
            self._info = ModelInfo(
                **{**self._info.__dict__, "cost_per_1k_output": cost_per_1k_output}
            )

        # HTTP client
        self.client = httpx.AsyncClient(timeout=timeout)

    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames using Modal-hosted model."""
        # Build request payload
        images = [frame.to_base64() for frame in frames[: self.max_images]]

        # Build prompt with context
        prompt_parts = []
        for msg in context:
            prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")

        user_text = self._build_user_content(frames, audio_transcript)
        prompt_parts.append(f"User: {user_text}")

        prompt = "\n".join(prompt_parts)

        # Build tools payload
        tools_payload = None
        if tools:
            tools_payload = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]

        payload = {
            "images": images,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tools": tools_payload,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Build headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Make request
        response = await self.client.post(
            self.endpoint_url,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()

        # Handle text response
        if "text" in result and result["text"]:
            yield Message(
                role="assistant",
                content=result["text"],
                timestamp=datetime.now(),
            )

        # Handle tool calls
        if "tool_calls" in result and result["tool_calls"]:
            for tc in result["tool_calls"]:
                yield ToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                    call_id=tc.get("id", ""),
                )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @property
    def supports_streaming(self) -> bool:
        # Modal endpoints typically don't stream
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
