"""Pure-stdlib perception layer — deterministic per-turn observations.

See :mod:`sherlock.perception.core` for the design rules. Off by default
(``PerceptionConfig.enabled = False``); when enabled, ``perceive()`` runs each
turn and ``render_observations()`` turns the result into the OBSERVED/PRIOR
slot block injected into LLM-1 (and, from Stage 2, LLM-3).
"""

from __future__ import annotations

from .core import Observation, perceive
from .render import render_observations

__all__ = ["Observation", "perceive", "render_observations"]
