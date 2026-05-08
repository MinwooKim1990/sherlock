"""litellm-backed provider. Covers Anthropic, OpenAI, Gemini, xAI, Ollama, LM Studio, etc."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import litellm

from sherlock.providers.base import BaseProvider, ChatMessage, ChatResponse, TokenUsage

if TYPE_CHECKING:
    from sherlock.config import ModelConfig

# Don't surface litellm's verbose logs by default.
litellm.suppress_debug_info = True


class LiteLLMProvider(BaseProvider):
    def __init__(self, model_config: "ModelConfig") -> None:
        self._cfg = model_config
        # Resolve API key from env var into litellm's expected env var name.
        # litellm reads e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
        # from the environment automatically. We only need to ensure it's set.
        key = model_config.resolved_api_key()
        if key:
            # Mirror the resolved value into the conventional name litellm expects.
            canonical = self._canonical_env_var(model_config.provider)
            if canonical and not os.environ.get(canonical):
                os.environ[canonical] = key

    @staticmethod
    def _canonical_env_var(provider: str) -> str | None:
        return {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "xai": "XAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "cohere": "COHERE_API_KEY",
        }.get(provider.lower())

    @property
    def model_id(self) -> str:
        return self._cfg.litellm_model_id()

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        litellm_messages = [m.to_dict() for m in messages]
        call_kwargs: dict = {
            "model": self.model_id,
            "messages": litellm_messages,
        }
        if self._cfg.api_base:
            call_kwargs["api_base"] = self._cfg.api_base
        call_kwargs.update(kwargs)

        resp = litellm.completion(**call_kwargs)
        content = resp.choices[0].message.content or ""
        usage_obj = getattr(resp, "usage", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        )
        cost = None
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None
        return ChatResponse(text=content, model=self.model_id, usage=usage, cost_usd=cost, raw=resp)

    async def achat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        litellm_messages = [m.to_dict() for m in messages]
        call_kwargs: dict = {
            "model": self.model_id,
            "messages": litellm_messages,
        }
        if self._cfg.api_base:
            call_kwargs["api_base"] = self._cfg.api_base
        call_kwargs.update(kwargs)

        resp = await litellm.acompletion(**call_kwargs)
        content = resp.choices[0].message.content or ""
        usage_obj = getattr(resp, "usage", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        )
        cost = None
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None
        return ChatResponse(text=content, model=self.model_id, usage=usage, cost_usd=cost, raw=resp)
