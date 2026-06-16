"""CallableProvider — wrap any user-supplied LLM call into a BaseProvider.

This is the "bring your own LLM" path: users hand Sherlock a function
that takes a list of messages and returns text. Anthropic, OpenAI,
litellm, Ollama, your own HTTP gateway — anything callable works.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Awaitable, Callable, Union

from sherlock.providers.base import BaseProvider, ChatMessage, ChatResponse, TokenUsage

# Three callable shapes accepted, in order of preference:
#   (a) f(messages: list[dict]) -> str
#   (b) f(messages: list[dict]) -> ChatResponse
#   (c) async f(messages: list[dict]) -> str | ChatResponse
ChatFn = Callable[[list[dict]], Union[str, ChatResponse, Awaitable[Union[str, ChatResponse]]]]


class CallableProvider(BaseProvider):
    """BaseProvider adapter for an arbitrary chat-completion callable.

    Example:
        import anthropic
        client = anthropic.Anthropic()

        def my_chat(messages: list[dict]) -> str:
            r = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=2048,
                messages=[m for m in messages if m["role"] != "system"],
                system="\\n".join(m["content"] for m in messages if m["role"] == "system"),
            )
            return r.content[0].text

        from sherlock import CallableProvider
        provider = CallableProvider(my_chat, model_id="anthropic/claude-haiku-4-5")

    Or, even simpler — go through `sherlock.create()`:

        agent = sherlock.create(main_chat=my_chat, system_prompt="You are helpful.")
        agent.chat("Hi")
    """

    def __init__(self, fn: ChatFn, *, model_id: str = "callable/user") -> None:
        self._fn = fn
        self._model_id = model_id
        # Detect async once at construction.
        self._is_async = inspect.iscoroutinefunction(fn)
        # v1.0 B5: pass prompt-cache hints ONLY to callables that declare a
        # `cache_hints` parameter — a plain f(messages) keeps receiving the
        # byte-identical payload it always has.
        self._accepts_cache_hints = False
        try:
            params = inspect.signature(fn).parameters
            self._accepts_cache_hints = "cache_hints" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except Exception:
            pass

    def _cache_hints(self, messages: list[ChatMessage]) -> dict | None:
        if not self._accepts_cache_hints:
            return None
        hints = {
            i: m.cache_stable_prefix_chars
            for i, m in enumerate(messages)
            if getattr(m, "cache_stable_prefix_chars", None)
        }
        return {"stable_prefix_chars": hints} if hints else None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _coerce(self, result: Union[str, ChatResponse]) -> ChatResponse:
        if isinstance(result, ChatResponse):
            return result
        if isinstance(result, str):
            return ChatResponse(
                text=result,
                model=self._model_id,
                usage=TokenUsage(),
                cost_usd=None,
                raw=None,
            )
        # Last-ditch: assume the caller's "response object" has a `.text` or `.content` attribute.
        text = getattr(result, "text", None) or getattr(result, "content", None) or str(result)
        return ChatResponse(text=str(text), model=self._model_id, usage=TokenUsage())

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        payload = [m.to_dict() for m in messages]
        hints = self._cache_hints(messages)
        extra = {"cache_hints": hints} if hints else {}
        if self._is_async:
            # Allow sync-callers to use an async function transparently.
            loop_result = asyncio.run(self._fn(payload, **extra))  # type: ignore[arg-type]
            return self._coerce(loop_result)
        result = self._fn(payload, **extra)  # type: ignore[misc]
        if inspect.iscoroutine(result):
            result = asyncio.run(result)
        return self._coerce(result)

    async def achat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        payload = [m.to_dict() for m in messages]
        hints = self._cache_hints(messages)
        extra = {"cache_hints": hints} if hints else {}
        if self._is_async:
            result = await self._fn(payload, **extra)  # type: ignore[misc]
            return self._coerce(result)
        # Sync callable — run in a thread so we don't block the event loop.
        result = await asyncio.to_thread(lambda: self._fn(payload, **extra))
        if inspect.iscoroutine(result):
            result = await result
        return self._coerce(result)
