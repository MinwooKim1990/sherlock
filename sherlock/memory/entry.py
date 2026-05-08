"""Memory entry model per SPEC §6.1.

Stored in SQLite via sqlmodel; the embedding lives in Chroma keyed by `id`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class MemoryType(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    SEARCH_RESULT = "search_result"
    TOOL_OUTPUT = "tool_output"
    USER_UTTERANCE = "user_utterance"
    SUMMARY = "summary"  # added to capture LLM-2 summaries (SPEC implies summary persistence)


class MemoryState(str, Enum):
    FRESH = "fresh"
    WARM = "warm"
    COLD = "cold"
    FORGOTTEN = "forgotten"


class MemorySource(str, Enum):
    USER = "user"
    LLM_INFERENCE = "llm_inference"
    SEARCH = "search"
    TOOL = "tool"
    SYSTEM = "system"  # for persona-summary-style records (T76 trap relevant)


class RLSignal(str, Enum):
    GOOD = "good"
    BAD = "bad"
    NEUTRAL = "neutral"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class MemoryEntry(SQLModel, table=True):
    """Persisted memory entry. Mirrors SPEC §6.1 with sqlmodel-compatible types."""

    __tablename__ = "memory_entry"

    id: str = Field(default_factory=_new_id, primary_key=True)
    conversation_id: str = Field(index=True)
    turn_id: Optional[str] = Field(default=None, index=True)

    type: MemoryType = Field(default=MemoryType.FACT, index=True)
    content: str  # raw text or JSON-encoded dict
    source: MemorySource = Field(default=MemorySource.USER, index=True)
    confidence: float = Field(default=1.0)

    semantic_triple_subject: Optional[str] = None
    semantic_triple_relation: Optional[str] = None
    semantic_triple_object: Optional[str] = None

    topic_cluster_id: Optional[str] = Field(default=None, index=True)
    state: MemoryState = Field(default=MemoryState.FRESH, index=True)

    pinned: bool = Field(default=False, index=True)

    created_at: datetime = Field(default_factory=_utcnow)
    last_used_at: datetime = Field(default_factory=_utcnow)
    use_count: int = Field(default=0)
    last_used_turn_index: int = Field(default=0, index=True)

    rl_signal: Optional[RLSignal] = None
    tags: str = Field(default="")  # comma-joined; vector layer handles richer tagging

    evidence: str = Field(default="")  # JSON-encoded list[str] for inferences

    def semantic_triple(self) -> Optional[tuple[str, str, str]]:
        if self.semantic_triple_subject and self.semantic_triple_relation:
            return (
                self.semantic_triple_subject,
                self.semantic_triple_relation,
                self.semantic_triple_object or "",
            )
        return None
