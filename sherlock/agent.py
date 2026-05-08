"""The Sherlock class — M1 surface.

M1: bare LLM-1 chat with no memory and no inference. Persists messages to
SQLite for later milestones to pick up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sherlock.config import Config
from sherlock.providers import BaseProvider, ChatMessage, ChatResponse, build_provider
from sherlock.storage import Conversation, Message, Storage


@dataclass
class TurnState:
    """Read-only snapshot of the last turn for inspection."""

    user_text: str
    response: ChatResponse
    messages_passed_to_llm1: list[ChatMessage]


class Sherlock:
    """M1 entry point. `chat()` is the synchronous, no-memory baseline."""

    def __init__(self, config: Config, *, provider: BaseProvider | None = None) -> None:
        self.config = config
        self._provider = provider or build_provider(config.models.main)
        self._storage = Storage(config.storage.sqlite_path)
        self._system_prompt = config.read_main_system_prompt()
        self._conversation: Conversation | None = None
        self._last_turn: TurnState | None = None

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    @property
    def conversation_id(self) -> Optional[str]:
        return self._conversation.id if self._conversation else None

    def _ensure_conversation(self) -> Conversation:
        if self._conversation is None:
            self._conversation = self._storage.create_conversation(project=self.config.project)
            # Persist the system prompt so a fresh process can replay the conversation.
            self._storage.add_message(
                self._conversation.id,
                role="system",
                content=self._system_prompt,
            )
        return self._conversation

    def _history_as_chat_messages(self) -> list[ChatMessage]:
        if self._conversation is None:
            return [ChatMessage(role="system", content=self._system_prompt)]
        msgs = self._storage.list_messages(self._conversation.id)
        return [ChatMessage(role=m.role, content=m.content) for m in msgs]

    def chat(self, user_input: str) -> str:
        """Send a user turn. Returns the assistant text. Persists everything."""
        conv = self._ensure_conversation()

        history = self._history_as_chat_messages()
        history.append(ChatMessage(role="user", content=user_input))

        # Persist the user turn before the LLM call so we don't lose it on a crash.
        self._storage.add_message(conv.id, role="user", content=user_input)

        response = self._provider.chat(history)

        self._storage.add_message(
            conv.id,
            role="assistant",
            content=response.text,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=response.cost_usd,
        )
        self._last_turn = TurnState(
            user_text=user_input,
            response=response,
            messages_passed_to_llm1=history,
        )
        return response.text

    def inspect_last_turn(self) -> TurnState | None:
        """Per SPEC § 8.1 `agent.inspect_last_turn()`."""
        return self._last_turn

    def messages(self) -> list[Message]:
        if self._conversation is None:
            return []
        return self._storage.list_messages(self._conversation.id)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Sherlock":
        return cls(Config.from_yaml(path))
