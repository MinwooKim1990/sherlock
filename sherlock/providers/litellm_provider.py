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
            # Open-source-model aggregators — names litellm reads for each.
            "deepinfra": "DEEPINFRA_API_KEY",
            "together": "TOGETHERAI_API_KEY",
            "together_ai": "TOGETHERAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(provider.lower())

    @property
    def model_id(self) -> str:
        return self._cfg.litellm_model_id()

    @staticmethod
    def _to_litellm_messages(messages: list[ChatMessage]) -> list[dict]:
        """v1.0 B5: a message carrying ``cache_stable_prefix_chars`` becomes
        OpenAI-format content blocks with ``cache_control`` on the stable
        block — litellm translates for Anthropic, drops it elsewhere. Plain
        messages stay plain dicts (byte-identical to before)."""
        out: list[dict] = []
        for m in messages:
            bps = getattr(m, "cache_breakpoints", None)
            if bps:
                # R11: N stability zones -> N cached blocks + a volatile tail.
                offsets = sorted({b for b in bps if 0 < b <= len(m.content)})[:3]
                blocks: list[dict] = []
                prev = 0
                for b in offsets:
                    blocks.append(
                        {
                            "type": "text",
                            "text": m.content[prev:b],
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
                    prev = b
                if prev < len(m.content):
                    blocks.append({"type": "text", "text": m.content[prev:]})
                if blocks:
                    out.append({"role": m.role, "content": blocks})
                    continue
            split = getattr(m, "cache_stable_prefix_chars", None)
            if split and split > 0:
                if split >= len(m.content):
                    out.append(
                        {
                            "role": m.role,
                            "content": [
                                {
                                    "type": "text",
                                    "text": m.content,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    )
                else:
                    out.append(
                        {
                            "role": m.role,
                            "content": [
                                {
                                    "type": "text",
                                    "text": m.content[:split],
                                    "cache_control": {"type": "ephemeral"},
                                },
                                {"type": "text", "text": m.content[split:]},
                            ],
                        }
                    )
            else:
                out.append(m.to_dict())
        return out

    @staticmethod
    def _cache_usage(usage_obj) -> tuple[int, int]:
        """(cache_read, cache_creation) from Anthropic- or OpenAI/Gemini-style
        usage. litellm surfaces these as attributes OR plain dicts depending on
        route/version — read both shapes."""

        def _field(obj, name):
            if obj is None:
                return 0
            if isinstance(obj, dict):
                return obj.get(name) or 0
            return getattr(obj, name, 0) or 0

        try:
            read = int(_field(usage_obj, "cache_read_input_tokens"))
            if not read:
                details = (
                    usage_obj.get("prompt_tokens_details")
                    if isinstance(usage_obj, dict)
                    else getattr(usage_obj, "prompt_tokens_details", None)
                )
                read = int(_field(details, "cached_tokens"))
            created = int(_field(usage_obj, "cache_creation_input_tokens"))
            return read, created
        except Exception:
            return 0, 0

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        litellm_messages = self._to_litellm_messages(messages)
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
        cache_read, cache_created = self._cache_usage(usage_obj)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_created,
        )
        cost = None
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None
        return ChatResponse(text=content, model=self.model_id, usage=usage, cost_usd=cost, raw=resp)

    async def achat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        litellm_messages = self._to_litellm_messages(messages)
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
        cache_read, cache_created = self._cache_usage(usage_obj)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_created,
        )
        cost = None
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None
        return ChatResponse(text=content, model=self.model_id, usage=usage, cost_usd=cost, raw=resp)
