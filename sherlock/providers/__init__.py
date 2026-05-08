"""Provider abstraction. M1 uses litellm under the hood (DEVIATION-003)."""
from sherlock.providers.base import BaseProvider, ChatMessage, ChatResponse
from sherlock.providers.fake import FakeProvider
from sherlock.providers.litellm_provider import LiteLLMProvider

__all__ = [
    "BaseProvider",
    "ChatMessage",
    "ChatResponse",
    "FakeProvider",
    "LiteLLMProvider",
    "build_provider",
]


def build_provider(model_config) -> BaseProvider:
    """Construct a provider from a `ModelConfig` (or `FakeProvider` if provider == 'fake')."""
    prov = model_config.provider.lower()
    if prov == "fake":
        return FakeProvider(model_id=model_config.model)
    return LiteLLMProvider(model_config=model_config)
