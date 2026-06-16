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
    DEEP_RESEARCH = "deep_research"  # v0.7: per-round deep-research session documents


class MemoryState(str, Enum):
    FRESH = "fresh"
    WARM = "warm"
    COLD = "cold"
    FORGOTTEN = "forgotten"


class MemorySource(str, Enum):
    USER = "user"
    LLM_INFERENCE = "llm_inference"
    LLM_2_PREDICTION = "llm_2_prediction"  # v0.4.0: forward-looking guesses
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
    # v0.9: turn at which the entry was FIRST created. Immutable — dedup
    # merges advance last_used_turn_index but never this.
    created_turn_index: int = Field(default=0, index=True)
    # v1.0: id of the entry that corrected this one (LLM-2 reconsolidation).
    # A superseded row is frozen: unpinned, skipped by dedup/decay/retrieval,
    # kept only as an audit trail.
    superseded_by: Optional[str] = Field(default=None, index=True)
    # v1.1 R34 (Graphiti-lite bi-temporal): the turn index at which this row
    # STOPPED being true — set by ``MemoryStore.supersede(..., turn_index=)``.
    # Together with created_turn_index this gives each fact a validity
    # interval, so "what was true before turn X?" is answerable via the
    # memory tools while invalid facts stay out of prompts (the existing
    # superseded_by exclusions already keep them out of retrieval).
    invalid_at_turn: Optional[int] = Field(default=None, index=True)
    # v1.0 B4: for SUMMARY entries — the last turn this summary covers. The
    # compaction frontier evicts raw turns ≤ this from the K-turn tail.
    summary_scope_to_turn: Optional[int] = Field(default=None, index=True)

    rl_signal: Optional[RLSignal] = None
    tags: str = Field(default="")  # comma-joined; vector layer handles richer tagging

    evidence: str = Field(default="")  # JSON-encoded list[str] for inferences

    # v0.5.0: sha256 of normalised content for O(1) indexed exact-dedup
    # (avoids scanning all same-type rows on every add).
    content_hash: str = Field(default="", index=True)

    @staticmethod
    def compute_hash(content: str) -> str:
        import hashlib

        norm = " ".join((content or "").strip().lower().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def semantic_triple(self) -> Optional[tuple[str, str, str]]:
        if self.semantic_triple_subject and self.semantic_triple_relation:
            return (
                self.semantic_triple_subject,
                self.semantic_triple_relation,
                self.semantic_triple_object or "",
            )
        return None


class MemoryLink(SQLModel, table=True):
    """v1.1 R33 (A-Mem-lite): an undirected similarity link between two
    memory entries in the same conversation.

    Created best-effort on ``MemoryStore.add`` when a genuinely-new entry
    has cosine ≥ 0.55 with an existing entry; consumed by the 1-hop
    expansion in ``HybridSearch.search``. ``src_id`` is the newer entry,
    ``dst_id`` the pre-existing neighbour; readers treat the edge as
    undirected (``MemoryStore.links_for`` returns both directions).
    """

    __tablename__ = "memory_link"

    id: str = Field(default_factory=_new_id, primary_key=True)
    src_id: str = Field(index=True)
    dst_id: str = Field(index=True)
    score: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=_utcnow)


class MemoryEntity(SQLModel, table=True):
    """v0.5.0 persistent entity index: (conversation, entity) → memory_id.

    Populated on `MemoryStore.add`; lets entity-lookup retrieval hit an
    index instead of scanning every memory in the conversation.
    """

    __tablename__ = "memory_entity"

    id: str = Field(default_factory=_new_id, primary_key=True)
    conversation_id: str = Field(index=True)
    entity: str = Field(index=True)
    memory_id: str = Field(index=True)
