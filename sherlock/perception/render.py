"""Render :class:`Observation` lists into the two-channel slot block.

OBSERVED facts are listed first (the model should trust them); PRIOR cues
follow, clearly labelled as probabilistic with explicit confidence, so a
guess can never be mistaken for a deterministic fact.
"""

from __future__ import annotations

from .core import Observation

_OBSERVED_HEADER = "OBSERVED (code-verified, deterministic — trust these):"
_PRIOR_HEADER = "PRIOR (probabilistic cues — NOT facts; confidence in parens):"


def render_observations(observations: list[Observation], *, max_observations: int = 12) -> str:
    """Return the slot text block, or ``""`` when there is nothing to say.

    OBSERVED observations get priority for the cap; PRIOR fills the remainder.
    """
    if not observations:
        return ""
    observed = [o for o in observations if o.channel == "observed"]
    prior = [o for o in observations if o.channel == "prior"]

    cap = max(1, int(max_observations))
    observed = observed[:cap]
    prior = prior[: max(0, cap - len(observed))]
    if not observed and not prior:
        return ""

    lines: list[str] = []
    if observed:
        lines.append(_OBSERVED_HEADER)
        lines.extend(f"- {o.text}" for o in observed)
    if prior:
        if lines:
            lines.append("")
        lines.append(_PRIOR_HEADER)
        for o in prior:
            c = f"(~{o.confidence:.1f}) " if o.confidence is not None else ""
            lines.append(f"- {c}{o.text}")
    return "\n".join(lines)
