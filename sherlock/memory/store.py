"""Combined SQLite (structured) + Chroma (vector) memory store.

SQLite holds the canonical record (per SPEC §6.1); Chroma holds the embedding
keyed by `MemoryEntry.id`. The two are kept in sync by `MemoryStore`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

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

# v1.1 R33: minimum cosine similarity for an automatic A-Mem-style link
# between a newly-added entry and an existing same-conversation entry.
LINK_SIM_THRESHOLD = 0.55
# v1.1 R33: a new entry links to at most this many nearest neighbours.
LINK_TOP_K = 3


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
        collection_name: str | None = None,
        redactor: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._engine = engine
        self._embed = embedding_provider
        # v0.5.0: optional secret/PII redactor applied to EVERY content write
        # (single choke point — covers user utterances, LLM-2 summary/facts,
        # LLM-3 inferences, and freshness search results). The raw transcript
        # is stored separately via Storage.add_message and stays faithful.
        self._redactor = redactor
        SQLModel.metadata.create_all(self._engine)
        # v0.5.0: add any newly-introduced columns to existing tables
        # (e.g. memory_entry.content_hash on a pre-v0.5 DB).
        try:
            from sherlock.storage.db import run_migrations

            run_migrations(self._engine)
        except Exception:
            pass

        self._vector_path = Path(vector_path)
        self._vector_path.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=str(self._vector_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        # v0.5.0: namespace the collection by embedder signature so that
        # switching embedders (e.g. fake dim=64 → local dim=384) uses a
        # FRESH collection instead of corrupting search with mismatched
        # dimensions. SQLite rows remain the source of truth; a stale
        # collection is simply unused.
        if collection_name is None:
            sig = getattr(embedding_provider, "collection_signature", "default")
            collection_name = f"sherlock_memories_{sig}"
        # v0.5.0: cosine distance — sentence embeddings (fastembed/openai)
        # aren't unit-normalised, so the default L2 metric conflates
        # magnitude with direction. Cosine is the correct similarity for
        # semantic recall. (Applies to newly-created collections; existing
        # ones keep their metric, which is fine — they're embedder-namespaced.)
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
        initial_state: MemoryState = MemoryState.FRESH,
    ) -> MemoryEntry:
        # v0.5.0 security: redact secrets/PII at the SINGLE write choke point,
        # before content feeds dedup-hash, the MemoryEntry row, the embedding
        # vector, the Chroma document, AND the entity index. This closes the
        # leak where a secret re-emitted by LLM-2 as a "fact" (or surfaced by
        # LLM-3 / freshness search) bypassed the user-utterance-only redaction.
        #
        # v0.5.1: redact EVERY string field, not just `content`. LLM-2/LLM-3 can
        # place a secret in evidence (json list), tags (topic/reasoning/freshness
        # label), or a semantic_triple element — all of which land in SQLite, the
        # entity index, and the eval/memory-tool output. Same single choke point.
        if self._redactor is not None:
            r = self._redactor
            try:
                if content:
                    content = r(content)
                if evidence:
                    evidence = r(evidence)
                if tags:
                    tags = r(tags)
                if semantic_triple:
                    semantic_triple = tuple((r(x) if x else x) for x in semantic_triple)
            except Exception:
                pass
        # Dedup-at-add: collapse re-emitted facts. LLM-2 tends to paraphrase
        # the same fact each summary cycle, so we use both:
        #   (a) exact / 60-char-prefix match (cheap, handles re-emits with
        #       slight tail variation),
        #   (b) embedding cosine similarity ≥ 0.92 against same-type entries
        #       (handles paraphrases like "Yujin has soba allergy" vs
        #       "User's child Yujin is allergic to soba").
        # The semantic pass only runs after (a) misses, and only against
        # the most recent 30 entries of the same type to keep it bounded.
        new_vec: list[float] | None = None
        if dedup and content.strip():
            norm = " ".join(content.strip().lower().split())
            chash = MemoryEntry.compute_hash(content)
            with Session(self._engine) as s:
                # v0.5.0: O(1) indexed exact-dedup across ALL history via
                # content_hash (was an O(n) scan of every same-type row).
                # v1.0: superseded rows are frozen — a correction's new text
                # must NEVER dedup-merge back into the hidden row it replaced
                # (the merged content would vanish from retrieval). All three
                # dedup scans skip them.
                exact = list(
                    s.exec(
                        select(MemoryEntry).where(
                            MemoryEntry.conversation_id == conversation_id,
                            MemoryEntry.type == type,
                            MemoryEntry.content_hash == chash,
                            MemoryEntry.superseded_by == None,  # noqa: E711
                        )
                    )
                )
                if exact:
                    existing = exact[0]
                    existing.use_count += 1
                    existing.last_used_at = _utcnow()
                    existing.last_used_turn_index = max(
                        existing.last_used_turn_index, last_used_turn_index
                    )
                    # v0.9: a restated fact is fresh again by definition —
                    # resurrect FORGOTTEN/COLD/WARM rows on dedup-merge.
                    if existing.state != MemoryState.FRESH:
                        existing.state = MemoryState.FRESH
                    if pinned and not existing.pinned:
                        existing.pinned = True
                    if confidence > existing.confidence:
                        existing.confidence = confidence
                    if existing.source != MemorySource.SYSTEM:
                        rank0 = {
                            MemorySource.USER: 4,
                            MemorySource.SEARCH: 3,
                            MemorySource.TOOL: 3,
                            MemorySource.LLM_INFERENCE: 2,
                            MemorySource.SYSTEM: 1,
                        }
                        if rank0.get(source, 0) > rank0.get(existing.source, 0):
                            existing.source = source
                    s.add(existing)
                    s.commit()
                    s.refresh(existing)
                    return existing

                # Fuzzy prefix + semantic passes: bound to the most RECENT
                # same-type rows (re-emits are near-term) so cost is O(40),
                # not O(all-history).
                stmt = (
                    select(MemoryEntry)
                    .where(
                        MemoryEntry.conversation_id == conversation_id,
                        MemoryEntry.type == type,
                        MemoryEntry.superseded_by == None,  # noqa: E711
                    )
                    .order_by(MemoryEntry.last_used_turn_index.desc())
                    .limit(40)
                )
                rows = list(s.exec(stmt))
                # Cheap prefix pass first.
                for existing in rows:
                    enorm = " ".join((existing.content or "").strip().lower().split())
                    if enorm and (enorm == norm or (len(enorm) > 30 and enorm[:60] == norm[:60])):
                        # Touch in-place; upgrade pinned/confidence if higher.
                        existing.use_count += 1
                        existing.last_used_at = _utcnow()
                        existing.last_used_turn_index = max(
                            existing.last_used_turn_index, last_used_turn_index
                        )
                        # v0.9: a restated fact is fresh again by definition.
                        if existing.state != MemoryState.FRESH:
                            existing.state = MemoryState.FRESH
                        # v0.9: prefix match with a DIFFERING tail is a
                        # correction, not a re-emit — newer information wins
                        # (Mem0-style update-on-conflict). Replace the stored
                        # content and every derived field (hash, embedding,
                        # Chroma document, entity index — synced after commit).
                        content_changed = enorm != norm
                        if content_changed:
                            existing.content = content
                            existing.content_hash = chash
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
                        if content_changed:
                            self._reembed(existing)
                        return existing

                # Semantic pass: embed the new content once, compare to the
                # most-recent 30 same-type rows. `rows` is ordered
                # last_used_turn_index DESC, so the most-recent are at the
                # FRONT — take rows[:30] (was rows[-30:], which compared the
                # OLDEST 30 and missed recent re-emits once history > 30).
                # Threshold 0.92 collapses paraphrases without merging genuinely
                # distinct facts.
                rows_recent = rows[:30]
                if rows_recent:
                    new_vec = self._embed.embed_one(content)
                    # Vector-store ids are kept in sync with SQLite ids; pull
                    # the embeddings via Chroma in one batch.
                    target_ids = [r.id for r in rows_recent]
                    try:
                        gres = self._collection.get(ids=target_ids, include=["embeddings"])
                    except Exception:
                        gres = None
                    if gres:
                        embs = gres.get("embeddings")
                        ids = gres.get("ids")
                        if embs is None:
                            embs = []
                        if ids is None:
                            ids = []
                        rows_by_id = {r.id: r for r in rows_recent}
                        for rid, ev in zip(ids, embs):
                            if ev is None or len(ev) == 0:
                                continue
                            sim = _cosine(new_vec, list(ev))
                            if sim >= 0.92:
                                existing = rows_by_id.get(rid)
                                if existing is None:
                                    continue
                                existing.use_count += 1
                                existing.last_used_at = _utcnow()
                                existing.last_used_turn_index = max(
                                    existing.last_used_turn_index, last_used_turn_index
                                )
                                # v0.9: restated fact is fresh again.
                                if existing.state != MemoryState.FRESH:
                                    existing.state = MemoryState.FRESH
                                if pinned and not existing.pinned:
                                    existing.pinned = True
                                if confidence > existing.confidence:
                                    existing.confidence = confidence
                                if existing.source != MemorySource.SYSTEM:
                                    rank2 = {
                                        MemorySource.USER: 4,
                                        MemorySource.SEARCH: 3,
                                        MemorySource.TOOL: 3,
                                        MemorySource.LLM_INFERENCE: 2,
                                        MemorySource.SYSTEM: 1,
                                    }
                                    if rank2.get(source, 0) > rank2.get(existing.source, 0):
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
            state=initial_state,
            pinned=pinned,
            evidence=evidence,
            tags=tags,
            last_used_turn_index=last_used_turn_index,
            # Creation turn == first-use turn; dedup merges never update it.
            created_turn_index=last_used_turn_index,
            content_hash=MemoryEntry.compute_hash(content),
        )
        with Session(self._engine) as s:
            s.add(entry)
            s.commit()
            s.refresh(entry)

        # Embed + write to Chroma. R33: reuse the vector the semantic-dedup
        # pass already computed (same redacted content) — never a second call.
        vec = new_vec if new_vec is not None else self._embed.embed_one(content)
        self._collection.add(
            ids=[entry.id],
            embeddings=[vec],
            metadatas=[
                {
                    "conversation_id": conversation_id,
                    "type": entry.type.value,
                    "source": entry.source.value,
                    "state": entry.state.value,
                    "pinned": entry.pinned,
                    "confidence": entry.confidence,
                }
            ],
            documents=[content],
        )
        # v0.5.0: populate the persistent entity index for fast lookup.
        self._index_entities(entry)
        # v1.1 R33: link the genuinely-new entry to its nearest neighbours.
        # Dedup merges never reach this point, so links only form on insert.
        self._link_similar(entry, vec)
        return entry

    def _link_similar(self, entry: "MemoryEntry", vec: list[float]) -> None:
        """v1.1 R33 (A-Mem-lite): link a freshly-inserted entry to the top-3
        most-similar same-conversation entries with cosine ≥ 0.55.

        Reuses ``vec`` (already computed for the Chroma write) — adds zero
        embedding calls. Best-effort: link failures never block the add.
        """
        try:
            from sherlock.memory.entry import MemoryLink

            res = self._collection.query(
                query_embeddings=[vec],
                # +1 because the new entry itself is already in the collection.
                n_results=LINK_TOP_K + 1,
                where={"conversation_id": entry.conversation_id},
                include=["embeddings"],
            )
            ids_outer = res.get("ids")
            embs_outer = res.get("embeddings")
            ids = ids_outer[0] if ids_outer is not None and len(ids_outer) else []
            embs = embs_outer[0] if embs_outer is not None and len(embs_outer) else []
            scored: list[tuple[str, float]] = []
            for rid, ev in zip(ids, embs):
                if rid == entry.id or ev is None or len(ev) == 0:
                    continue
                # Exact cosine in Python — independent of the collection metric.
                sim = _cosine(vec, list(ev))
                if sim >= LINK_SIM_THRESHOLD:
                    scored.append((rid, sim))
            if not scored:
                return
            scored.sort(key=lambda t: t[1], reverse=True)
            with Session(self._engine) as s:
                for rid, sim in scored[:LINK_TOP_K]:
                    s.add(MemoryLink(src_id=entry.id, dst_id=rid, score=float(sim)))
                s.commit()
        except Exception:
            pass

    def links_for(self, entry_id: str) -> list[tuple[str, float]]:
        """v1.1 R33: neighbours linked to ``entry_id`` (either direction),
        strongest first. Returns ``[(other_entry_id, link_score), ...]``."""
        from sherlock.memory.entry import MemoryLink

        out: list[tuple[str, float]] = []
        with Session(self._engine) as s:
            for ln in s.exec(select(MemoryLink).where(MemoryLink.src_id == entry_id)):
                out.append((ln.dst_id, ln.score))
            for ln in s.exec(select(MemoryLink).where(MemoryLink.dst_id == entry_id)):
                out.append((ln.src_id, ln.score))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def _reembed(self, entry: "MemoryEntry") -> None:
        """Re-sync the derived stores after an in-place content update
        (v0.9 prefix-dedup correction): upsert the Chroma embedding/document
        and rebuild the entry's entity-index rows. Best-effort, like the
        other vector-side mirrors."""
        try:
            vec = self._embed.embed_one(entry.content)
            self._collection.upsert(
                ids=[entry.id],
                embeddings=[vec],
                metadatas=[
                    {
                        "conversation_id": entry.conversation_id,
                        "type": entry.type.value,
                        "source": entry.source.value,
                        "state": entry.state.value,
                        "pinned": entry.pinned,
                        "confidence": entry.confidence,
                    }
                ],
                documents=[entry.content],
            )
        except Exception:
            pass
        try:
            from sherlock.memory.entry import MemoryEntity

            with Session(self._engine) as s:
                for er in s.exec(select(MemoryEntity).where(MemoryEntity.memory_id == entry.id)):
                    s.delete(er)
                s.commit()
        except Exception:
            pass
        self._index_entities(entry)

    def _index_entities(self, entry: "MemoryEntry") -> None:
        """Extract entities from an entry and write index rows. Best-effort."""
        try:
            from sherlock.memory.entry import MemoryEntity
            from sherlock.rag.hybrid import _entry_entity_pool

            tokens = _entry_entity_pool(entry)
            if not tokens:
                return
            with Session(self._engine) as s:
                for tok in tokens:
                    s.add(
                        MemoryEntity(
                            conversation_id=entry.conversation_id,
                            entity=tok,
                            memory_id=entry.id,
                        )
                    )
                s.commit()
        except Exception:
            pass

    def find_by_entities(self, conversation_id: str, tokens: set[str]) -> list[MemoryEntry]:
        """Return memories whose indexed entities intersect ``tokens``.

        Indexed lookup (O(matches)), replacing the prior O(all-rows) scan.
        """
        if not tokens:
            return []
        from sherlock.memory.entry import MemoryEntity

        toks = {t.lower() for t in tokens}
        with Session(self._engine) as s:
            idx_rows = list(
                s.exec(
                    select(MemoryEntity).where(
                        MemoryEntity.conversation_id == conversation_id,
                        MemoryEntity.entity.in_(toks),
                    )
                )
            )
            mem_ids = list({r.memory_id for r in idx_rows})
            if not mem_ids:
                return []
            rows = list(s.exec(select(MemoryEntry).where(MemoryEntry.id.in_(mem_ids))))
        return rows

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
            self._collection.update(ids=[memory_id], metadatas=[{"state": new_state.value}])
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

    def cap_pinned(self, conversation_id: str, max_pinned: int = 25) -> int:
        """Hard cap on PIN count per conversation. Demote the least-valuable
        pins first. PROTECTED from demotion (v0.5.0):
          - SYSTEM-source pins (persona notes / domain hints)
          - USER-source pins (facts the user explicitly stated)
          - persona-summary entries
        Only inferred/search/tool pins are eligible for demotion. If the
        protected set alone exceeds the cap, nothing is demoted (the cap is
        soft for high-value pins). Returns number of items demoted.
        """
        from sherlock.memory.entry import MemorySource

        with Session(self._engine) as s:
            stmt = select(MemoryEntry).where(
                MemoryEntry.conversation_id == conversation_id,
                MemoryEntry.pinned == True,  # noqa: E712
            )
            pinned = list(s.exec(stmt))
            # v0.9: DEEP_RESEARCH docs are pinned for durability only — they
            # are read on demand and never injected into the prompt slot, so
            # they neither count against the cap nor get demoted by it.
            # v1.0: superseded rows are frozen and never count either.
            pinned = [
                p
                for p in pinned
                if p.type not in (MemoryType.DEEP_RESEARCH, MemoryType.DEEP_RESEARCH_RAW)
                and not p.superseded_by
            ]
            if len(pinned) <= max_pinned:
                return 0

            def _protected(p) -> bool:
                if p.source in (MemorySource.SYSTEM, MemorySource.USER):
                    return True
                if "persona_summary" in (p.tags or ""):
                    return True
                return False

            demotable = [p for p in pinned if not _protected(p)]
            # Demote lowest-confidence, oldest-touched first.
            demotable.sort(key=lambda p: (p.confidence, p.last_used_turn_index, p.created_at))
            demoted = 0
            target_demote = len(pinned) - max_pinned
            for entry in demotable:
                if demoted >= target_demote:
                    break
                entry.pinned = False
                s.add(entry)
                demoted += 1
            s.commit()
        return demoted

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

    def set_summary_scope(self, memory_id: str, scope_to_turn: int) -> None:
        """v1.0 B4: record the last turn a SUMMARY covers (frontier marker).
        Monotonic — a dedup-merged summary keeps its widest coverage."""
        try:
            with Session(self._engine) as s:
                row = s.get(MemoryEntry, memory_id)
                if row is None:
                    return
                cur = row.summary_scope_to_turn or 0
                row.summary_scope_to_turn = max(cur, int(scope_to_turn))
                s.add(row)
                s.commit()
        except Exception:
            pass

    def supersede(self, old_id: str, new_id: str, turn_index: int | None = None) -> None:
        """v1.0: mark ``old_id`` as corrected by ``new_id`` (non-destructive).

        The old row keeps its content as an audit trail but is unpinned and
        frozen — excluded from dedup, retrieval, cap_pinned and decay.

        v1.1 R34 (bi-temporal): when ``turn_index`` is given, it is recorded
        as ``invalid_at_turn`` — the turn at which the fact stopped being
        true — so deliberate recall can answer "what was true before X?".
        """
        with Session(self._engine) as s:
            row = s.get(MemoryEntry, old_id)
            if not row:
                return
            row.superseded_by = new_id
            row.pinned = False
            if turn_index is not None:
                row.invalid_at_turn = turn_index
            s.add(row)
            s.commit()
        try:
            self._collection.update(ids=[old_id], metadatas=[{"pinned": False}])
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
        from sherlock.memory.entry import MemoryEntity, MemoryLink

        with Session(self._engine) as s:
            row = s.get(MemoryEntry, memory_id)
            if not row:
                return
            s.delete(row)
            # v0.5.1: cascade the entity index too. Without this, repeated
            # deletes (persona-summary replacement every summary cycle, decay
            # eviction) leave orphaned MemoryEntity rows that accumulate and
            # could surface stale memory_ids in find_by_entities.
            for er in s.exec(select(MemoryEntity).where(MemoryEntity.memory_id == memory_id)):
                s.delete(er)
            # v1.1 R33: cascade memory links (either endpoint).
            for ln in s.exec(select(MemoryLink).where(MemoryLink.src_id == memory_id)):
                s.delete(ln)
            for ln in s.exec(select(MemoryLink).where(MemoryLink.dst_id == memory_id)):
                s.delete(ln)
            s.commit()
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass

    def delete_conversation_memories(self, conversation_id: str) -> int:
        """Hard-delete every memory entry attached to a conversation.

        Cascade helper for session deletion. Also clears the matching
        vector entries from Chroma.  Returns the count removed.
        """
        from sherlock.memory.entry import MemoryEntity, MemoryLink

        ids: list[str] = []
        with Session(self._engine) as s:
            rows = list(
                s.exec(select(MemoryEntry).where(MemoryEntry.conversation_id == conversation_id))
            )
            for r in rows:
                ids.append(r.id)
                s.delete(r)
            # Cascade: remove entity-index rows too.
            for er in s.exec(
                select(MemoryEntity).where(MemoryEntity.conversation_id == conversation_id)
            ):
                s.delete(er)
            # v1.1 R33: links are conversation-internal (both endpoints live
            # in this conversation) — cascade them by endpoint id.
            if ids:
                for ln in s.exec(select(MemoryLink).where(MemoryLink.src_id.in_(ids))):
                    s.delete(ln)
                for ln in s.exec(select(MemoryLink).where(MemoryLink.dst_id.in_(ids))):
                    s.delete(ln)
            s.commit()
        if ids:
            try:
                self._collection.delete(ids=ids)
            except Exception:
                pass
        return len(ids)

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

    def latest_retrieval_keywords(self, conversation_id: str) -> str:
        """v1.0: content of the rolling LLM-2 ``retrieval_keywords`` entry
        (a single replace-in-place row written by the summarizer), or ``""``.
        """
        rows = [
            e
            for e in self.list(conversation_id=conversation_id)
            if "retrieval_keywords" in (e.tags or "")
        ]
        if not rows:
            return ""
        rows.sort(key=lambda e: e.last_used_turn_index, reverse=True)
        return rows[0].content

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
            rows = {e.id: e for e in s.exec(select(MemoryEntry).where(MemoryEntry.id.in_(ids)))}

        scored: list[tuple[MemoryEntry, float]] = []
        for mid, dist in zip(ids, distances):
            entry = rows.get(mid)
            if not entry:
                continue
            # v1.0: superseded rows are frozen (audit trail only); the rolling
            # retrieval_keywords entry is plumbing, not a recallable memory.
            if entry.superseded_by:
                continue
            if "retrieval_keywords" in (entry.tags or ""):
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
