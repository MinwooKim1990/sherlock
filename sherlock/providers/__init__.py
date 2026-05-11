"""Provider abstraction.

- CallableProvider: bring-your-own-LLM (wraps any callable).
- LiteLLM-backed (DEVIATION-003) for the spec'd path.
- WrapperProvider (DEVIATION-004) for subscription-auth paths.
- FakeProvider for hermetic tests.
"""
from sherlock.providers.base import BaseProvider, ChatMessage, ChatResponse
from sherlock.providers.callable_provider import CallableProvider
from sherlock.providers.fake import FakeProvider
from sherlock.providers.litellm_provider import LiteLLMProvider
from sherlock.providers.wrapper_provider import WrapperProvider

__all__ = [
    "BaseProvider",
    "CallableProvider",
    "ChatMessage",
    "ChatResponse",
    "FakeProvider",
    "LiteLLMProvider",
    "WrapperProvider",
    "build_provider",
]


def build_provider(model_config) -> BaseProvider:
    prov = model_config.provider.lower()
    if prov == "fake":
        return FakeProvider(model_id=model_config.model)
    if prov.startswith("wrapper"):
        return WrapperProvider(model_config=model_config)
    return LiteLLMProvider(model_config=model_config)
