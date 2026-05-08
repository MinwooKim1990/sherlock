"""M4 RAG layer (light): hybrid vector + BM25, with reranking-by-fusion.

The full M4 reranker (Cohere / bge) is deferred; this scope is the minimum
to beat vector-only search per SPEC §7.1.
"""
from sherlock.rag.hybrid import HybridSearch

__all__ = ["HybridSearch"]
