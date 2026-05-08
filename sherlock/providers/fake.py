"""Deterministic in-process provider for hermetic tests.

The fake echoes the last user message with a small annotation so tests can
verify message-routing without hitting any external API.
"""
from __future__ import annotations

from sherlock.providers.base import BaseProvider, ChatMessage, ChatResponse, TokenUsage


class FakeProvider(BaseProvider):
    def __init__(self, model_id: str = "fake/echo", canned_reply: str | None = None) -> None:
        self._model_id = model_id
        self._canned_reply = canned_reply

    @property
    def model_id(self) -> str:
        return self._model_id

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        if self._canned_reply is not None:
            text = self._canned_reply
        else:
            last_user = next(
                (m for m in reversed(messages) if m.role == "user"),
                None,
            )
            user_text = last_user.content if last_user else ""
            text = f"[fake:{self._model_id}] you said: {user_text}"
        prompt_chars = sum(len(m.content) for m in messages)
        return ChatResponse(
            text=text,
            model=self._model_id,
            usage=TokenUsage(
                prompt_tokens=prompt_chars // 4,
                completion_tokens=len(text) // 4,
                total_tokens=(prompt_chars + len(text)) // 4,
            ),
            cost_usd=0.0,
            raw=None,
        )
