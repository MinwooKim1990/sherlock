"""Combined SQLite (structured) + Chroma (vector) memory store.

SQLite holds the canonical record (per SPEC §6.1); Chroma holds the embedding
keyed by `MemoryEntry.id`. The two are kept in sync by `MemoryStore`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sqlmodel import Session, SQLModel, select

from sherlock.memory.embeddings import EmbeddingProvider
from sherlock.memory.entry import (
    MemoryEntry,
    MemorySource,
    MemoryState,
    MemoryType,
    RLSignal,
    _utcnow,
)


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


class MemoryStore:
    """Per-conversation memory store. Composes the global SQLite engine + a Chroma collection."""

    def __init__(
        self,
        engine,
        embedding_provider: EmbeddingProvider,
        vector_path: str | Path,
        collection_name: str = "sherlock_memories",
    ) -> None:
        self._engine = engine
        self._embed = embedding_provider
        SQLModel.metadata.create_all(self._engine)

        self._vector_path = Path(vector_path)
        self._vector_path.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=str(self._vector_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._chroma.get_or_create_collection(name=collection_name)

    # ---------------- write path ----------------

    def add(
        self,
        *,
        conversation_id: str,
        content: str,
        type: MemoryType = MemoryType.FACT,
        source: MemorySource = MemorySource.USER,
        confidence: float = 1.0,
        turn_id: Optional[str] = None,
        last_used_turn_index: int = 0,
        pinned: bool = False,
        topic_cluster_id: Optional[str] = None,
        evidence: str = "",
        tags: str = "",
        semantic_triple: Optional[tuple[str, str, str]] = None,
        dedup: bool = True,
    ) -> MemoryEntry:
        # Dedup-at-add: collapse re-emitted facts (LLM-2 tends to re-emit the
        # same pinned fact on every summary cycle). When an existing entry
        # has the same conversation_id + type + normalised content, just
        # touch + (optionally) upgrade pinned/confidence and return it.
        if dedup and content.strip():
            norm = " ".join(content.strip().lower().split())
            with Session(self._engine) as s:
                stmt = select(MemoryEntry).where(
                    MemoryEntry.conversation_id == conversation_id,
                    MemoryEntry.type == type,
                )
                for existing in s.exec(stmt):
                    enorm = " ".join((existing.content or "").strip().lower().split())
                    if enorm and (enorm == norm or (len(enorm) > 30 and enorm[:60] == norm[:60])):
                        # Touch in-place; upgrade pinned/confidence if higher.
                        existing.use_count += 1
                        existing.last_used_at = _utcnow()
                        existing.last_used_turn_index = max(
                            existing.last_used_turn_index, last_used_turn_index
                        )
                        if pinned and not existing.pinned:
                            existing.pinned = True
                        if confidence > existing.confidence:
                            existing.confidence = confidence
                        # Source: SYSTEM is sticky. A persona-note fact must NEVER
                        # become user-stated just because LLM-2 paraphrased it back
                        # with source="user". This was the loop-3 regression.
                        if existing.source != MemorySource.SYSTEM:
                            rank = {
                                MemorySource.USER: 4,
                                MemorySource.SEARCH: 3,
                                MemorySource.TOOL: 3,
                                MemorySource.LLM_INFERENCE: 2,
                                MemorySource.SYSTEM: 1,
                            }
                            if rank.get(source, 0) > rank.get(existing.source, 0):
                                existing.source = source
                        s.add(existing)
                        s.commit()
                        s.refresh(existing)
                        return existing
        st_subject, st_relation, st_object = (None, None, None)
        if semantic_triple:
            st_subject, st_relation, st_object = semantic_triple

        entry = MemoryEntry(
            conversation_id=conversation_id,
            turn_id=turn_id,
            type=type,
            content=content,
            source=source,
            confidence=confidence,
            semantic_triple_subject=st_subject,
            semantic_triple_relation=st_relation,
            semantic_triple_object=st_object,
            topic_cluster_id=topic_cluster_id,
            state=MemoryState.FRESH,
            pinned=pinned,
            evidence=evidence,
            tags=tags,
            last_used_turn_index=last_used_turn_index,
        )
        with Session(self._engine) as s:
            s.add(entry)
            s.commit()
            s.refresh(entry)

        # Embed + write to Chroma.
        vec = self._embed.embed_one(content)
        self._collection.add(
            ids=[entry.id],
            embeddings=[vec],
            metadatas=[{
                "conversation_id": conversation_id,
                "type": entry.type.value,
                "source": entry.source.value,
                "state": entry.state.value,
                "pinned": entry.pinned,
                "confidence": entry.confidence,
            }],
            documents=[content],
        )
        return entry

    def update_state(self, memory_id: str, new_state: MemoryState) -> None:
        with Session(self._engine) as s:
            row = s.get(MemoryEntry, memory_id)
            if not row:
                return
            row.state = new_state
            s.add(row)
            s.commit()
        # Mirror in Chroma metadata.
        try:
            self._collection.update(
                ids=[memory_id], metadatas=[{"state": new_state.value}]
            )
        except Exception:
            pass

    def touch(self, memory_id: str, turn_index: int) -> None:
        with Session(self._engine) as s:
            row = s.get(MemoryEntry, memory_id)
            if not row:
                return
            row.use_count += 1
            row.last_used_at = _utcnow()
            row.last_used_turn_index = turn_index
            if row.state == MemoryState.WARM:
                row.state = MemoryState.FRESH
            elif row.state == MemoryState.COLD:
                row.state = MemoryState.WARM
            s.add(row)
            s.commit()

    def pin(self, memory_id: str, value: bool = True) -> None:
        with Session(self._engine) as s:
            row = s.get(MemoryEntry, memory_id)
            if not row:
                return
            row.pinned = value
            s.add(row)
            s.commit()
        try:
            self._collection.update(ids=[memory_id], metadatas=[{"pinned": value}])
        except Exception:
            pass

    def set_rl_signal(self, memory_id: str, signal: RLSignal) -> None:
        with Session(self._engine) as s:
            row = s.get(MemoryEntry, memory_id)
            if not row:
                return
            row.rl_signal = signal
            s.add(row)
            s.commit()

    def soft_delete(self, memory_id: str) -> None:
        self.update_state(memory_id, MemoryState.FORGOTTEN)

    def hard_delete(self, memory_id: str) -> None:
        with Session(self._engine) as s:
            row = s.get(MemoryEntry, memory_id)
            if not row:
                return
            s.delete(row)
            s.commit()
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass

    # ---------------- read path ----------------

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        with Session(self._engine) as s:
            return s.get(MemoryEntry, memory_id)

    def list(
        self,
        *,
        conversation_id: Optional[str] = None,
        state: Optional[MemoryState] = None,
        source: Optional[MemorySource] = None,
        pinned: Optional[bool] = None,
    ) -> list[MemoryEntry]:
        with Session(self._engine) as s:
            stmt = select(MemoryEntry)
            if conversation_id is not None:
                stmt = stmt.where(MemoryEntry.conversation_id == conversation_id)
            if state is not None:
                stmt = stmt.where(MemoryEntry.state == state)
            if source is not None:
                stmt = stmt.where(MemoryEntry.source == source)
            if pinned is not None:
                stmt = stmt.where(MemoryEntry.pinned == pinned)
            return list(s.exec(stmt))

    def search(
        self,
        query: str,
        *,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        include_states: tuple[MemoryState, ...] = (
            MemoryState.FRESH,
            MemoryState.WARM,
            MemoryState.COLD,
        ),
        confidence_threshold: float = 0.0,
        exclude_inferences_below: float | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return (entry, score) pairs sorted by score desc."""
        if not query.strip():
            return []
        qvec = self._embed.embed_one(query)
        where: dict = {}
        if conversation_id is not None:
            where["conversation_id"] = conversation_id

        # Query Chroma.
        results = self._collection.query(
            query_embeddings=[qvec],
            n_results=max(top_k * 4, 8),
            where=where if where else None,
        )
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not ids:
            return []

        # Pull rows.
        with Session(self._engine) as s:
            rows = {
                e.id: e
                for e in s.exec(select(MemoryEntry).where(MemoryEntry.id.in_(ids)))
            }

        scored: list[tuple[MemoryEntry, float]] = []
        for mid, dist in zip(ids, distances):
            entry = rows.get(mid)
            if not entry:
                continue
            if entry.state not in include_states and not entry.pinned:
                continue
            if entry.confidence < confidence_threshold:
                continue
            if (
                exclude_inferences_below is not None
                and entry.type == MemoryType.INFERENCE
                and entry.confidence < exclude_inferences_below
            ):
                continue
            # Chroma cosine distance ∈ [0, 2]; smaller = more similar. Convert
            # to a similarity in [0, 1].
            similarity = 1.0 - (dist / 2.0)
            scored.append((entry, similarity))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]

    def cosine_between(self, content_a: str, content_b: str) -> float:
        a, b = self._embed.embed([content_a, content_b])
        return _cosine(a, b)

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embed

    # ---------------- bulk helpers used by decay ----------------

    def all_for_conversation(self, conversation_id: str) -> list[MemoryEntry]:
        return self.list(conversation_id=conversation_id)
