"""M2 Memory layer.

Components:
- `entry.py`: SPEC §6.1 Memory entry model.
- `embeddings.py`: embedding provider abstraction (litellm).
- `store.py`: combined SQLite + Chroma vector store.
- `decay.py`: 4-state lifecycle (fresh / warm / cold / forgotten).
- `summarizer.py`: LLM-2 summarization cycle.
- `k_turn.py`: K-turn original retention policy.
"""

from sherlock.memory.decay import DecayEngine, DecayConfig
from sherlock.memory.embeddings import EmbeddingProvider, build_embedding_provider
from sherlock.memory.entry import MemoryEntry, MemoryState, MemoryType
from sherlock.memory.k_turn import KTurnPolicy
from sherlock.memory.store import MemoryStore
from sherlock.memory.summarizer import SummarizerConfig, SummarizerEngine

__all__ = [
    "DecayConfig",
    "DecayEngine",
    "EmbeddingProvider",
    "KTurnPolicy",
    "MemoryEntry",
    "MemoryState",
    "MemoryStore",
    "MemoryType",
    "SummarizerConfig",
    "SummarizerEngine",
    "build_embedding_provider",
]
