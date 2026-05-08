"""WrapperProvider — runtime provider backed by `cli-wrapper-unified`.

Subscription-auth path. Used when API keys are not reachable from this
Python process (see DEVIATION-004 in INTENT_DEVIATIONS.md).

The wrapper supports `provider/model` strings via `create("claude" | "codex" | "gemini")`
with subscription auth + automatic API-key fallback.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from sherlock.providers.base import BaseProvider, ChatMessage, ChatResponse, TokenUsage

if TYPE_CHECKING:
    from sherlock.config import ModelConfig


class WrapperProvider(BaseProvider):
    """Map our `ModelConfig` to a unified_cli client.

    Provider field convention:
      provider="wrapper"            → fall back to claude (haiku) by default
      provider="wrapper-claude"     → wrapper backed by Claude
      provider="wrapper-codex"      → wrapper backed by Codex
      provider="wrapper-gemini"     → wrapper backed by Gemini

    Or simpler: provider="anthropic"/"openai"/"gemini" as usual but with
    `model` set to a wrapper-known id (e.g. "haiku" / "gpt-5.4-mini" /
    "gemini-3.1-flash-lite-preview") and an optional `via_wrapper: true`
    flag — but to keep config explicit we use the provider="wrapper-*" form.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
        self._cfg = model_config
        prov = model_config.provider.lower()
        # Translate to unified_cli's provider name.
        if prov in ("wrapper", "wrapper-claude", "anthropic-wrapper"):
            self._wrapped = "claude"
        elif prov in ("wrapper-codex", "openai-wrapper"):
            self._wrapped = "codex"
        elif prov in ("wrapper-gemini", "gemini-wrapper"):
            self._wrapped = "gemini"
        else:
            self._wrapped = "claude"

        from unified_cli import create

        # `create` returns a per-provider client object with .chat(prompt, ...).
        # Timeout=300s (vs wrapper default 120s) — the consolidator pass for
        # an 80-turn replay produces a ~74KB prompt → claude takes ~150s. Default
        # 120s timed out in Loop 17/18 → bulletproof fallback fired → 12-13/100.
        self._client = create(self._wrapped, model=model_config.model, timeout=300.0)

    @property
    def model_id(self) -> str:
        return f"{self._wrapped}/{self._cfg.model}"

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        # The wrapper's chat() is single-prompt, not multi-turn. Flatten our
        # message list into a single prompt that preserves role markers.
        prompt = self._flatten(messages)
        try:
            r = self._client.chat(prompt)
        except Exception as exc:
            return ChatResponse(
                text=f"[wrapper-error: {type(exc).__name__}: {exc}]",
                model=self.model_id,
                usage=TokenUsage(),
                cost_usd=0.0,
                raw=None,
            )
        text = getattr(r, "text", "") or ""
        usage_obj = getattr(r, "usage", None)
        usage = TokenUsage()
        if usage_obj is not None:
            usage = TokenUsage(
                prompt_tokens=getattr(usage_obj, "input_tokens", 0)
                or getattr(usage_obj, "prompt_tokens", 0)
                or 0,
                completion_tokens=getattr(usage_obj, "output_tokens", 0)
                or getattr(usage_obj, "completion_tokens", 0)
                or 0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return ChatResponse(text=text, model=self.model_id, usage=usage, cost_usd=0.0, raw=r)

    @staticmethod
    def _flatten(messages: list[ChatMessage]) -> str:
        # Hard guard against the wrapper's underlying CLI invoking tools
        # (WebFetch / Write / Edit) — we want a TEXT-ONLY response. The
        # subscription-auth claude/codex CLIs ship with tool access on by
        # default; this banner sits at the top of every flattened prompt.
        guard = (
            "[SYSTEM — SHERLOCK WRAPPER GUARD]\n"
            "Respond with TEXT ONLY. Do not invoke any tools. Do not "
            "create, edit, or write any files on the filesystem. Do not "
            "fetch URLs. Do not search the web. Just produce the answer "
            "as plain text or strict JSON depending on the role's prompt."
        )
        parts: list[str] = [guard]
        for m in messages:
            role_label = {
                "system": "[SYSTEM]",
                "user": "[USER]",
                "assistant": "[ASSISTANT]",
            }.get(m.role, f"[{m.role.upper()}]")
            parts.append(f"{role_label}\n{m.content}")
        # Trailing nudge so the wrapper completes a fresh assistant turn.
        parts.append("[ASSISTANT]")
        return "\n\n".join(parts)
