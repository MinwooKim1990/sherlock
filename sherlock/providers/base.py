"""Provider ABC + message/response dataclasses.

The ABC is intentionally thin so a hand-rolled per-provider implementation
can replace `LiteLLMProvider` without touching call sites.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float | None = None
    raw: object = None  # provider-native response, when available


class BaseProvider(ABC):
    """Synchronous-and-async chat completion interface.

    M1 uses the synchronous path. M5 will switch the pipeline to `achat`
    for parallel companion execution.
    """

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse: ...

    async def achat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Async path. Default implementation falls back to sync via thread."""
        import asyncio

        return await asyncio.to_thread(self.chat, messages, **kwargs)
