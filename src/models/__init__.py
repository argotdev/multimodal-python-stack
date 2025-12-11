"""Vision-language model providers."""

from src.models.base import VisionLanguageModel, ModelInfo
from src.models.openai import OpenAIVisionModel
from src.models.anthropic import AnthropicVisionModel
from src.models.google import GoogleVisionModel
from src.models.groq import GroqVisionModel
from src.models.fireworks import FireworksVisionModel
from src.models.together import TogetherVisionModel
from src.models.modal import ModalVisionModel

__all__ = [
    "VisionLanguageModel",
    "ModelInfo",
    "OpenAIVisionModel",
    "AnthropicVisionModel",
    "GoogleVisionModel",
    "GroqVisionModel",
    "FireworksVisionModel",
    "TogetherVisionModel",
    "ModalVisionModel",
    "create_model",
    "list_models",
]

# Provider registry
PROVIDERS = {
    "openai": OpenAIVisionModel,
    "anthropic": AnthropicVisionModel,
    "google": GoogleVisionModel,
    "groq": GroqVisionModel,
    "fireworks": FireworksVisionModel,
    "together": TogetherVisionModel,
    "modal": ModalVisionModel,
}


def create_model(provider: str, model_id: str | None = None, **kwargs) -> VisionLanguageModel:
    """Factory function to create a vision-language model.

    Args:
        provider: Provider name (openai, anthropic, google, groq, fireworks, together)
        model_id: Specific model ID (uses provider default if not specified)
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Configured VisionLanguageModel instance

    Example:
        model = create_model("openai", "gpt-4o-mini")
        model = create_model("anthropic")  # Uses default model
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {', '.join(PROVIDERS.keys())}"
        )

    model_class = PROVIDERS[provider]

    if model_id:
        return model_class(model_id=model_id, **kwargs)
    else:
        return model_class(**kwargs)


def list_models() -> dict[str, list[ModelInfo]]:
    """List all available models by provider.

    Returns:
        Dictionary mapping provider names to lists of ModelInfo
    """
    result = {}
    for provider, model_class in PROVIDERS.items():
        result[provider] = model_class.available_models()
    return result
