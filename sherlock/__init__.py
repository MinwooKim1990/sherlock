"""Project Sherlock — domain-agnostic context-curation library.

Quick start (bring your own LLM):

    from sherlock import Sherlock

    def my_llm(messages: list[dict]) -> str:
        # call any LLM — anthropic, openai, ollama, your own gateway, etc.
        # `messages` is list of {"role": "system"|"user"|"assistant", "content": "..."}
        return "..."

    agent = Sherlock.with_callable(
        main_chat=my_llm,
        system_prompt="You are a helpful assistant.",
    )
    reply = agent.chat("Hello")

Sherlock manages conversation history, runs LLM-2 (compaction) and
LLM-3 (Sherlock-style inference) in the background, and curates a
durable provenance-aware memory store. Your LLM decides when to invoke
the companions by emitting an optional `<<sherlock-companions: compact,
infer>>` tag at the end of its reply (the tag is stripped before the
user sees it). If your LLM never emits the tag, Sherlock will still
fire on the final turn so memory is never empty.

Spec-driven YAML path (advanced):

    from sherlock import Sherlock, Config
    config = Config.from_yaml("sherlock.yaml")
    agent = Sherlock(config)

See `sherlock.example.yaml` for the YAML schema.
"""

from sherlock.agent import Sherlock, TurnState
from sherlock.config import Config
from sherlock.providers import (
    BaseProvider,
    CallableProvider,
    ChatMessage,
    ChatResponse,
    FakeProvider,
)

__version__ = "0.2.0"
__all__ = [
    "BaseProvider",
    "CallableProvider",
    "ChatMessage",
    "ChatResponse",
    "Config",
    "FakeProvider",
    "Sherlock",
    "TurnState",
    "__version__",
]
