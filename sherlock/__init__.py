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
durable provenance-aware memory store. Your LLM can invoke the
companions explicitly by emitting a `<<sherlock-companions: compact,
infer>>` tag at the end of its reply (the tag is stripped before the
user sees it). As a safety net so neither stays dormant when your LLM
under-emits the tag, Sherlock also auto-fires `compact` every N turns
and `infer` selectively on a topic shift (config `memory.auto_infer`:
"smart" default | "off" | "always").

Spec-driven YAML path (advanced):

    from sherlock import Sherlock
    agent = Sherlock.from_yaml("sherlock.yaml")

Use `Sherlock.from_yaml(...)`, NOT the bare `Sherlock(config)` constructor:
`from_yaml` also wires the companion prompts (LLM-2/LLM-3) and the web-search
engines from the config. The bare constructor leaves those unset (LLM-1 only).
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

__version__ = "1.11.0"
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
