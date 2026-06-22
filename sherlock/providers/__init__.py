"""Provider abstraction.

- CallableProvider: bring-your-own-LLM (wraps any callable).
- LiteLLM-backed for the spec'd path.
- WrapperProvider for subscription-auth paths.
- FakeProvider for hermetic tests.
"""

from sherlock.providers.base import BaseProvider, ChatMessage, ChatResponse
from sherlock.providers.callable_provider import CallableProvider
from sherlock.providers.fake import FakeProvider

# NOTE: LiteLLMProvider / WrapperProvider are imported LAZILY (inside
# build_provider and __getattr__ below). Importing litellm eagerly on
# `import sherlock` triggers a network cost-map fetch + slow import, which
# is wasteful for the common CallableProvider / local paths. (v0.5.0)

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


def __getattr__(name: str):
    # PEP 562 module-level lazy attribute access — keeps
    # `from sherlock.providers import LiteLLMProvider` working without an
    # eager litellm import at package load.
    if name == "LiteLLMProvider":
        from sherlock.providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider
    if name == "WrapperProvider":
        from sherlock.providers.wrapper_provider import WrapperProvider

        return WrapperProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_provider(model_config) -> BaseProvider:
    prov = model_config.provider.lower()
    if prov == "fake":
        return FakeProvider(model_id=model_config.model)
    if prov.startswith("wrapper"):
        from sherlock.providers.wrapper_provider import WrapperProvider

        return WrapperProvider(model_config=model_config)
    from sherlock.providers.litellm_provider import LiteLLMProvider

    return LiteLLMProvider(model_config=model_config)
