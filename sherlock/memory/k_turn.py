"""K-turn original-retention policy per SPEC §5.1.

The most recent K turns are passed to LLM 1 uncompressed. This module
simply computes K — actual injection happens in the slot assembler.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KTurnPolicy:
    k_min: int = 3
    k_max: int = 10
    adaptive: bool = True
    context_utilisation_high: float = 0.70

    def k(self, *, topic_changed: bool, context_utilisation: float) -> int:
        if not self.adaptive:
            return self.k_min
        if topic_changed:
            return self.k_min
        if context_utilisation >= self.context_utilisation_high:
            return self.k_min
        return self.k_max
