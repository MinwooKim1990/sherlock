"""Hybrid vector + BM25 search with score fusion (Reciprocal Rank Fusion).

Vector ranks come from `MemoryStore.search`. BM25 is computed on-the-fly
over the candidate set to keep things stateless. RRF score:

    rrf_score(d) = Σ_r  1 / (k + rank_r(d))

where k = 60 (Cormack et al., 2009, the standard).
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from sherlock.memory.entry import MemoryEntry, MemoryState
from sherlock.memory.store import MemoryStore


def _tokenise(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


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
    ) -> list[tuple[MemoryEntry, float]]:
        if not query.strip():
            return []

        # Vector hits.
        vec_hits = self.store.search(
            query,
            conversation_id=conversation_id,
            top_k=top_k * 4,
            include_states=include_states,
            confidence_threshold=confidence_threshold,
            exclude_inferences_below=exclude_inferences_below,
        )
        if not vec_hits:
            return []

        # BM25 over the same candidate pool: this keeps our BM25 index
        # session-local instead of building a second persistent index.
        candidates = [e for e, _ in vec_hits]
        corpus = [_tokenise(e.content) for e in candidates]
        if not corpus:
            return vec_hits[:top_k]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(_tokenise(query))
        bm25_ranked = sorted(
            range(len(candidates)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_ranked)}

        # Vector ranks.
        vec_rank = {i: i for i in range(len(candidates))}

        # RRF combine.
        fused_score: dict[int, float] = {}
        for i in range(len(candidates)):
            score = 0.0
            score += 1.0 / (self.rrf_k + vec_rank[i] + 1)
            score += 1.0 / (self.rrf_k + bm25_rank[i] + 1)
            fused_score[i] = score

        ranked_indices = sorted(fused_score.keys(), key=lambda i: fused_score[i], reverse=True)
        return [(candidates[i], fused_score[i]) for i in ranked_indices[:top_k]]
