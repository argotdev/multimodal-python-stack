"""Agent configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""

    # Frame processing
    max_frames: int = 4  # Maximum frames to send per request
    frame_batch_size: int = 1  # Process after this many frames
    frame_interval_ms: int = 1000  # Minimum ms between frame captures

    # Audio processing
    min_audio_chars: int = 50  # Process after this many transcribed chars
    audio_chunk_seconds: float = 5.0  # Audio chunk duration

    # Context management
    max_context_tokens: int = 4000  # Maximum tokens in context window
    max_context_messages: int = 20  # Maximum messages to keep

    # Model defaults
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"

    # System prompt
    system_prompt: str = ""

    # Timeouts
    request_timeout_seconds: float = 30.0
    tool_timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls, **overrides: Any) -> AgentConfig:
        """Create config from environment variables with optional overrides."""
        env_values = {
            "max_frames": int(os.getenv("AGENT_MAX_FRAMES", "4")),
            "frame_batch_size": int(os.getenv("AGENT_FRAME_BATCH_SIZE", "1")),
            "frame_interval_ms": int(os.getenv("AGENT_FRAME_INTERVAL_MS", "1000")),
            "min_audio_chars": int(os.getenv("AGENT_MIN_AUDIO_CHARS", "50")),
            "max_context_tokens": int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", "4000")),
            "default_provider": os.getenv("AGENT_DEFAULT_PROVIDER", "openai"),
            "default_model": os.getenv("AGENT_DEFAULT_MODEL", "gpt-4o-mini"),
        }
        env_values.update(overrides)
        return cls(**env_values)


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""

    api_key: str
    base_url: str | None = None
    organization: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def get_provider_config(provider: str) -> ProviderConfig:
    """Get configuration for a specific provider from environment."""
    configs = {
        "openai": ProviderConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            organization=os.getenv("OPENAI_ORG_ID"),
        ),
        "anthropic": ProviderConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        ),
        "google": ProviderConfig(
            api_key=os.getenv("GOOGLE_API_KEY", ""),
        ),
        "groq": ProviderConfig(
            api_key=os.getenv("GROQ_API_KEY", ""),
        ),
        "fireworks": ProviderConfig(
            api_key=os.getenv("FIREWORKS_API_KEY", ""),
        ),
        "together": ProviderConfig(
            api_key=os.getenv("TOGETHER_API_KEY", ""),
        ),
    }
    if provider not in configs:
        raise ValueError(f"Unknown provider: {provider}")
    return configs[provider]
