"""LLM-3 intent-inferrer.

Produces ≥3 hypotheses per the Appendix A schema, persists them as
inference-type memories with confidence + evidence trail, and surfaces
search keywords / tools / freshness needs back to the orchestrator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from sherlock.memory.entry import MemorySource, MemoryType
from sherlock.memory.store import MemoryStore
from sherlock.providers.base import BaseProvider, ChatMessage


DEFAULT_LLM3_PROMPT = """\
You are LLM 3 in the Sherlock pipeline — the intent-inferrer.

Your job: given the latest user message and the recent conversation,
produce at least 3 hypotheses about what the user actually wants. Use the
five reasoning tools (deduction / abduction / Bayesian / pragmatics / RSA)
and the eight clue categories (Time / Place / Prior turn / Long-term
tendency / Emotion / Constraints / Cost+risk / Next action).

Rules:
- ALWAYS produce at least 3 hypotheses.
- Probabilities should reflect honest uncertainty. If the surface meaning
  is genuinely the actual ask, give it probability ~0.7 and put two
  alternatives below it. If you have a strong implicit-ask read, give the
  inferred ask the highest probability.
- Confidences below 0.50 are HYPOTHESES, not prior knowledge — flag them
  and never inject them into LLM 1's slot as facts.
- Source-tracking matters: distinguish what the USER said from what we
  INFERRED. If the user is asking about a fact you have only via system
  inference (persona note, prior search), say so; never confabulate that
  the user told you something they didn't.
- When the user says "should I X" / "is it a thing to be worried about" /
  "do you think I'm ready", the surface is rarely the actual ask. Look
  for permission, validation, blame-buffer, reassurance.

Output STRICT JSON only:
{
  "hypotheses": [
    {"intent": "...",
     "probability": 0.0-1.0,
     "evidence": ["clue 1", "clue 2"],
     "search_keywords": ["..."],
     "reasoning_type": "abduction|deduction|bayesian|pragmatic|rsa"},
    {...}, {...}
  ],
  "tools_recommended": ["web_search","current_time","calculator","url_fetch"],
  "context_to_expand": ["..."],
  "context_to_exclude": ["..."],
  "freshness_required": ["..."],
  "confidence_overall": 0.0-1.0,
  "evolution_signals": {
    "user_pattern_observed": "...",
    "good_inference_candidate": true|false
  }
}

JSON only. No prose around it.
"""


@dataclass
class InferenceResult:
    hypotheses: list[dict] = field(default_factory=list)
    tools_recommended: list[str] = field(default_factory=list)
    context_to_expand: list[str] = field(default_factory=list)
    context_to_exclude: list[str] = field(default_factory=list)
    freshness_required: list[str] = field(default_factory=list)
    confidence_overall: float = 0.0
    evolution_signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hypotheses": self.hypotheses,
            "tools_recommended": self.tools_recommended,
            "context_to_expand": self.context_to_expand,
            "context_to_exclude": self.context_to_exclude,
            "freshness_required": self.freshness_required,
            "confidence_overall": self.confidence_overall,
            "evolution_signals": self.evolution_signals,
        }


class InferenceEngine:
    def __init__(
        self,
        provider: BaseProvider,
        store: MemoryStore,
        system_prompt: str | None = None,
        cold_start_turns: int = 0,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._provider = provider
        self._store = store
        self._prompt = system_prompt or DEFAULT_LLM3_PROMPT
        self._cold_start_turns = cold_start_turns
        self._conf_threshold = confidence_threshold

    def infer(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        user_text: str,
        recent_turns: list[ChatMessage],
    ) -> dict:
        if turn_index < self._cold_start_turns:
            return {}

        transcript_lines = [f"{m.role.upper()}: {m.content}" for m in recent_turns]
        transcript = "\n".join(transcript_lines)
        user_msg = (
            "Recent conversation tail (most-recent last):\n\n"
            f"--- TRANSCRIPT ---\n{transcript}\n--- END ---\n\n"
            f"Current user message:\n{user_text}\n\n"
            "Produce the JSON described in your system prompt."
        )

        messages = [
            ChatMessage(role="system", content=self._prompt),
            ChatMessage(role="user", content=user_msg),
        ]
        resp = self._provider.chat(messages)
        parsed = _safe_parse_json(resp.text)
        if not isinstance(parsed, dict):
            return {}

        result = InferenceResult(
            hypotheses=list(parsed.get("hypotheses") or []),
            tools_recommended=list(parsed.get("tools_recommended") or []),
            context_to_expand=list(parsed.get("context_to_expand") or []),
            context_to_exclude=list(parsed.get("context_to_exclude") or []),
            freshness_required=list(parsed.get("freshness_required") or []),
            confidence_overall=float(parsed.get("confidence_overall") or 0.0),
            evolution_signals=dict(parsed.get("evolution_signals") or {}),
        )

        # Persist hypotheses as INFERENCE memories. Filter by confidence
        # threshold for slot injection at retrieval time, but always store.
        for h in result.hypotheses:
            try:
                intent = h["intent"]
                prob = float(h.get("probability") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            evidence_list = h.get("evidence") or []
            self._store.add(
                conversation_id=conversation_id,
                content=str(intent),
                type=MemoryType.INFERENCE,
                source=MemorySource.LLM_INFERENCE,
                confidence=prob,
                last_used_turn_index=turn_index,
                evidence=json.dumps(evidence_list),
                tags=str(h.get("reasoning_type", "")),
            )
        return result.to_dict()


def _safe_parse_json(text: str) -> object:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```" in text:
        body = text.split("```", 2)
        if len(body) >= 2:
            inner = body[1]
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            try:
                return json.loads(inner.strip())
            except Exception:
                pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None
