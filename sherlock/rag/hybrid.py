"""Hybrid vector + BM25 search with score fusion (Reciprocal Rank Fusion).

Vector ranks come from `MemoryStore.search`. BM25 is computed on-the-fly
over the candidate set to keep things stateless. RRF score:

    rrf_score(d) = Σ_r  1 / (k + rank_r(d))

where k = 60 (Cormack et al., 2009, the standard).

v0.4.0 — entity-indexed Tier 2 retrieval. Before RRF runs, we extract
proper-noun-style entities from the query and pull every memory whose
``tags`` field, semantic_triple subject/object, or content contains a
matching token. These deterministic matches get rank-0 weight in RRF,
so they dominate semantic noise on factual recall queries
(e.g. "Yujin 알레르기").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from sherlock.memory.entry import MemoryEntry, MemoryState
from sherlock.memory.store import MemoryStore

# Stopwords are stripped from BM25 tokenisation so that function words
# ("for", "my", "what", "is", "는", "이") don't create spurious lexical
# matches that bury a strong semantic (vector) hit. (v0.5.0 fix)
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
he her his how i if in into is it its me my no not of on or our she so some that
the their them then there these they this to too us was we were what when where
which who why will with would you your about
은 는 이 가 을 를 의 에 와 과 도 만 으로 로 에서 한 그 저 나 너 우리
""".split())


# v1.0: Korean is agglutinative — "유진이는" never whitespace-matches the
# stored "유진", so the BM25 channel was blind to particle-suffixed recall
# queries. For Hangul runs of length ≥3 we ADDITIONALLY emit character
# bigrams alongside the original token ("유진이는" → 유진/진이/이는); a
# 2-char run's only bigram IS the token, so those stay as-is. ASCII-only
# tokens are untouched.
_HANGUL_RUN_RE = re.compile(r"[가-힣]{3,}")


def _tokenise(text: str) -> list[str]:
    toks = [t.strip(".,!?;:\"'()[]") for t in text.lower().split()]
    out: list[str] = []
    for t in toks:
        if not t or t in _STOPWORDS:
            continue
        out.append(t)
        for run in _HANGUL_RUN_RE.findall(t):
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


# Entities = capitalised tokens (English proper nouns) + hangul-only
# tokens of length ≥2 (Korean names / places). Note we deliberately
# avoid `\b` because Python's word-boundary treats Hangul as a word
# character, so `Yujin은` wouldn't match. Instead we use a negative
# lookbehind on alphanumerics.
_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Z][A-Za-z0-9]{1,}"  # CapitalisedWord (Latin)
    r"|[가-힣]{2,}"  # Korean word
)


def extract_entities(text: str) -> set[str]:
    """Best-effort entity extraction from a query.

    Returns a set of lowercase tokens (English) and as-is tokens (Korean).
    Keeps the heuristic simple — over-extraction is fine because the
    matcher only boosts memories that also contain the entity.
    """
    if not text:
        return set()
    toks = set()
    for m in _ENTITY_RE.findall(text):
        if len(m) < 2:
            continue
        toks.add(m.lower() if m[0].isascii() else m)
    return toks


def _entry_entity_pool(entry: MemoryEntry) -> set[str]:
    """All tokens that count as 'this entry is about' for entity overlap."""
    pool: set[str] = set()
    # Semantic triple subject / object — highest-quality signal.
    if entry.semantic_triple_subject:
        pool.add(entry.semantic_triple_subject.lower())
    if entry.semantic_triple_object:
        pool.add(entry.semantic_triple_object.lower())
    # Tags (comma-joined).
    for t in (entry.tags or "").split(","):
        t = t.strip()
        if t:
            pool.add(t.lower())
    # Content fallback — any Capitalised / Korean token.
    pool |= extract_entities(entry.content or "")
    return pool


@dataclass
class HybridSearch:
    store: MemoryStore
    rrf_k: int = 60

    def search(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        top_k: int = 5,
        include_states: tuple[MemoryState, ...] = (
            MemoryState.FRESH,
            MemoryState.WARM,
            MemoryState.COLD,
        ),
        confidence_threshold: float = 0.0,
        exclude_inferences_below: float | None = None,
        current_turn_index: int | None = None,
        max_per_type: dict | None = None,
        expand_links: bool = True,
    ) -> list[tuple[MemoryEntry, float]]:
        if not query.strip():
            return []

        # Tier 2: entity-indexed deterministic lookup. Always runs first.
        # When entities are found, we pull every memory in this
        # conversation tagged with one of them — bypassing similarity
        # scoring entirely. These get rank-0 in the RRF below.
        entity_tokens = extract_entities(query)
        entity_hits: list[MemoryEntry] = []
        if entity_tokens and conversation_id is not None:
            # v0.5.0: indexed entity lookup (was an O(all-rows) scan).
            # Fall back to a scan only if the index method is unavailable.
            candidates = []
            finder = getattr(self.store, "find_by_entities", None)
            if callable(finder):
                candidates = finder(conversation_id, entity_tokens)
            else:  # pragma: no cover - legacy fallback
                candidates = self.store.list(conversation_id=conversation_id)
            seen_ids: set[str] = set()
            for e in candidates:
                if e.state == MemoryState.FORGOTTEN:
                    continue
                # v1.0: superseded rows are frozen (corrected facts must not
                # resurface via the entity channel); the rolling
                # retrieval_keywords entry is plumbing, never a RAG result.
                if e.superseded_by:
                    continue
                if "retrieval_keywords" in (e.tags or ""):
                    continue
                if e.confidence < confidence_threshold:
                    continue
                if e.id in seen_ids:
                    continue
                # Verify overlap (index may be coarser than the live pool).
                if _entry_entity_pool(e) & entity_tokens:
                    entity_hits.append(e)
                    seen_ids.add(e.id)

        # Vector hits (Tier 4 fallback).
        vec_hits = self.store.search(
            query,
            conversation_id=conversation_id,
            top_k=top_k * 4,
            include_states=include_states,
            confidence_threshold=confidence_threshold,
            exclude_inferences_below=exclude_inferences_below,
        )
        if not vec_hits and not entity_hits:
            return []

        # Union the candidate pool: entity matches first (rank 0), then
        # vector matches in their existing order. Dedupe by id.
        seen: set[str] = set()
        candidates: list[MemoryEntry] = []
        for e in entity_hits:
            if e.id in seen:
                continue
            candidates.append(e)
            seen.add(e.id)
        for e, _ in vec_hits:
            if e.id in seen:
                continue
            candidates.append(e)
            seen.add(e.id)
        if not candidates:
            return []

        # BM25 over the same candidate pool: this keeps our BM25 index
        # session-local instead of building a second persistent index.
        corpus = [_tokenise(e.content) for e in candidates]
        bm25 = BM25Okapi(corpus) if any(corpus) else None
        if bm25 is not None:
            bm25_scores = list(bm25.get_scores(_tokenise(query)))
            bm25_ranked = sorted(
                range(len(candidates)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )
            bm25_rank = {idx: rank for rank, idx in enumerate(bm25_ranked)}
        else:
            bm25_scores = [0.0] * len(candidates)
            bm25_rank = {i: len(candidates) for i in range(len(candidates))}

        # Vector ranks: entity hits are at indices [0, len(entity_hits))
        # — give them rank 0 in the vector channel as a precision boost.
        n_entity = len(entity_hits)
        vec_rank = {i: (0 if i < n_entity else i) for i in range(len(candidates))}

        # Fusion (v0.5.0 — fixed). The VECTOR channel is the semantic
        # backbone and must dominate: a strong vector match must NOT be
        # buried by BM25 noise on a keyword-free query (the bug that made
        # "what foods are unsafe for my child" rank the peanut-allergy fact
        # last). BM25 only contributes for candidates with REAL lexical
        # overlap (bm25_score > 0) and is down-weighted relative to vector.
        VEC_WEIGHT = 2.0
        BM25_WEIGHT = 1.0
        fused_score: dict[int, float] = {}
        for i in range(len(candidates)):
            score = VEC_WEIGHT / (self.rrf_k + vec_rank[i] + 1)
            if bm25_scores[i] > 0:  # gate: no lexical overlap → no BM25 term
                score += BM25_WEIGHT / (self.rrf_k + bm25_rank[i] + 1)
            if i < n_entity:
                score += 0.5  # entity-precision boost (deterministic match)
            fused_score[i] = score

        # v1.1 R29 — composite retrieval scoring (generative-agents style).
        # Two SMALL additive boosts on top of fusion, gated behind
        # current_turn_index so legacy callers (None) get byte-identical
        # scores and ordering:
        #   recency    = 0.2 * exp(-age/40)        max 0.2 at age 0,
        #                                          ~0.074 at 40 turns
        #   importance = min(use_count, 10) * 0.01 max 0.1
        # Worst-case combined boost is 0.3 < the 0.5 entity boost (which
        # every candidate is equally eligible for on top), so entity-first
        # ordering is preserved.
        if current_turn_index is not None:
            for i, e in enumerate(candidates):
                age = max(0, current_turn_index - (e.created_turn_index or 0))
                fused_score[i] += 0.2 * math.exp(-age / 40.0)
                fused_score[i] += min(e.use_count or 0, 10) * 0.01

        ranked_indices = sorted(fused_score.keys(), key=lambda i: fused_score[i], reverse=True)

        # v1.1 R30 — per-type result caps. Keys may be MemoryType or its
        # string value; types absent from the dict are uncapped. Overflow is
        # dropped lowest-ranked first (we walk in rank order), and the freed
        # slots backfill from the next-ranked candidates.
        type_counts: dict[str, int] = {}
        caps: dict[str, int] = {}
        if max_per_type:
            caps = {getattr(k, "value", k): v for k, v in max_per_type.items()}
            kept: list[int] = []
            for i in ranked_indices:
                tval = getattr(candidates[i].type, "value", str(candidates[i].type))
                cap = caps.get(tval)
                if cap is not None:
                    if type_counts.get(tval, 0) >= cap:
                        continue
                    type_counts[tval] = type_counts.get(tval, 0) + 1
                kept.append(i)
            ranked_indices = kept

        results = [(candidates[i], fused_score[i]) for i in ranked_indices[:top_k]]
        # Cap counts must reflect only what actually made the cut.
        if caps:
            type_counts = {}
            for e, _ in results:
                tval = getattr(e.type, "value", str(e.type))
                type_counts[tval] = type_counts.get(tval, 0) + 1

        # v1.1 R33 — 1-hop link expansion (A-Mem style). For the top-3 direct
        # hits, append linked entries not already in the results with
        # score = 0.3 * link_score (≤ 0.3, and appended AFTER the direct hits
        # so they never outrank them). Capped at +3 appended entries; the
        # standard exclusions (superseded / FORGOTTEN / retrieval_keywords)
        # and the R30 per-type caps still apply.
        links_fn = getattr(self.store, "links_for", None)
        if expand_links and results and callable(links_fn):
            existing_ids = {e.id for e, _ in results}
            appended = 0
            for seed, _ in list(results[:3]):
                if appended >= 3:
                    break
                try:
                    links = links_fn(seed.id)
                except Exception:
                    links = []
                for other_id, link_score in links:
                    if appended >= 3:
                        break
                    if other_id in existing_ids:
                        continue
                    other = self.store.get(other_id)
                    if other is None:
                        continue
                    if other.state == MemoryState.FORGOTTEN:
                        continue
                    if other.superseded_by:
                        continue
                    if "retrieval_keywords" in (other.tags or ""):
                        continue
                    if other.confidence < confidence_threshold:
                        continue
                    tval = getattr(other.type, "value", str(other.type))
                    cap = caps.get(tval)
                    if cap is not None and type_counts.get(tval, 0) >= cap:
                        continue
                    results.append((other, 0.3 * link_score))
                    existing_ids.add(other_id)
                    if cap is not None:
                        type_counts[tval] = type_counts.get(tval, 0) + 1
                    appended += 1

        return results
