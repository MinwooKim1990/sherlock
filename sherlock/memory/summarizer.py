"""LLM-2 summarization cycle per SPEC § 4.2 (async turn preparation).

M2-scope responsibilities:
  - decide whether a summary pass is warranted (n-turn or topic-change trigger)
  - call LLM 2 with the bootstrap-authored summary prompt (or a default)
  - parse the JSON output
  - write summary + extracted facts to the memory store
  - propose retrieval keywords for next turn

The summary prompt is intentionally small here; a richer prompt is authored
by the Bootstrap engine in M3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from sherlock.providers.base import BaseProvider, ChatMessage
from sherlock.memory.entry import MemorySource, MemoryType
from sherlock.memory.store import MemoryStore


DEFAULT_LLM2_PROMPT = """\
You are LLM 2 in the Sherlock pipeline. Your job is to compress a chunk of
recent conversation into structured memory.

Given the recent turns, output JSON with these fields exactly:
{
  "summary": "<2-4 sentence dense prose summary>",
  "facts": [
    {"content": "<one fact>", "type": "fact|inference|user_utterance",
     "source": "user|llm_inference|search|tool|system",
     "confidence": 0.0-1.0,
     "semantic_triple": ["subject", "relation", "object"] or null,
     "evidence": ["short clue"...] or [],
     "pin_recommended": true|false,
     "let_fade": true|false}
  ],
  "topic_label": "<short 2-4 word topic>",
  "topic_changed_from_previous": true|false,
  "retrieval_keywords": ["next-turn lookup keyword"...]
}

Rules:
- Pin only facts the user clearly wants permanently remembered (location,
  role, key dates, constraints). Default pin=false.
- Mark let_fade=true for offhand mentions that are not referenced again
  (cafes, books, podcasts mentioned once with "anyway" pivots).
- Inferences carry source="llm_inference" and confidence < 1.0.
- User utterances carry source="user" and confidence = 1.0.
- Never invent facts. If a fact is implied, mark it as inference, not user.

Output JSON only — no prose around it.
"""


@dataclass
class SummarizerConfig:
    trigger_every_n_turns: int = 3
    trigger_on_topic_change: bool = True
    topic_change_similarity_threshold: float = 0.4
    prompt: str = DEFAULT_LLM2_PROMPT


class SummarizerEngine:
    def __init__(
        self,
        provider: BaseProvider,
        store: MemoryStore,
        config: SummarizerConfig | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._cfg = config or SummarizerConfig()

    def should_run(
        self,
        *,
        turn_index: int,
        prev_user_text: str | None,
        current_user_text: str,
    ) -> tuple[bool, bool]:
        """Return (should_run, topic_changed)."""
        if turn_index <= 0:
            return False, False
        topic_changed = False
        if prev_user_text and self._cfg.trigger_on_topic_change:
            sim = self._store.cosine_between(prev_user_text, current_user_text)
            topic_changed = sim < self._cfg.topic_change_similarity_threshold
        n_turn_trigger = (turn_index % self._cfg.trigger_every_n_turns) == 0
        return (n_turn_trigger or topic_changed), topic_changed

    def run(
        self,
        *,
        conversation_id: str,
        recent_turns: list[ChatMessage],
        turn_index: int,
    ) -> dict:
        """Call LLM 2 over the recent-turn window. Returns the parsed JSON dict."""
        # Build the user message for LLM 2: a compact transcript.
        transcript_lines = []
        for m in recent_turns:
            transcript_lines.append(f"{m.role.upper()}: {m.content}")
        transcript = "\n".join(transcript_lines)
        user_msg = (
            "Here is the most recent stretch of conversation. Produce the "
            "JSON described in your system prompt.\n\n--- TRANSCRIPT ---\n"
            f"{transcript}\n--- END TRANSCRIPT ---"
        )

        messages = [
            ChatMessage(role="system", content=self._cfg.prompt),
            ChatMessage(role="user", content=user_msg),
        ]
        resp = self._provider.chat(messages)

        parsed = _safe_parse_json(resp.text)
        if not isinstance(parsed, dict):
            return {
                "summary": resp.text.strip()[:500],
                "facts": [],
                "topic_label": None,
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
            }

        # Persist the summary itself as a memory entry.
        if parsed.get("summary"):
            self._store.add(
                conversation_id=conversation_id,
                content=parsed["summary"],
                type=MemoryType.SUMMARY,
                source=MemorySource.LLM_INFERENCE,
                confidence=0.9,
                last_used_turn_index=turn_index,
                tags=parsed.get("topic_label", "") or "",
            )

        # Persist each extracted fact.
        for fact in parsed.get("facts", []):
            try:
                content = fact["content"]
            except (KeyError, TypeError):
                continue
            ftype = _coerce_type(fact.get("type"))
            fsrc = _coerce_source(fact.get("source"))
            confidence = float(fact.get("confidence") or 1.0)
            triple = fact.get("semantic_triple")
            triple_tuple = None
            if isinstance(triple, list) and len(triple) == 3:
                triple_tuple = (str(triple[0]), str(triple[1]), str(triple[2]))
            evidence_list = fact.get("evidence") or []
            self._store.add(
                conversation_id=conversation_id,
                content=str(content),
                type=ftype,
                source=fsrc,
                confidence=confidence,
                last_used_turn_index=turn_index,
                pinned=bool(fact.get("pin_recommended")),
                evidence=json.dumps(evidence_list),
                semantic_triple=triple_tuple,
            )
            # `let_fade` shows up in M2-style decay nudge: leave the entry as
            # FRESH for one turn but the decay engine will move it warm→cold
            # quickly because it's not used again. Captured in evidence too.

        return parsed


def _safe_parse_json(text: str) -> object:
    """Try hard to parse JSON out of an LLM response that may have prose around it."""
    text = text.strip()
    if not text:
        return None
    # Try direct.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced code blocks.
    if "```" in text:
        body = text.split("```", 2)
        if len(body) >= 2:
            inner = body[1]
            if inner.startswith("json\n") or inner.startswith("json "):
                inner = inner[5:]
            try:
                return json.loads(inner.strip())
            except Exception:
                pass
    # Try to grab the largest {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def _coerce_type(v: object) -> MemoryType:
    if isinstance(v, MemoryType):
        return v
    s = str(v or "").lower()
    try:
        return MemoryType(s)
    except Exception:
        return MemoryType.FACT


def _coerce_source(v: object) -> MemorySource:
    if isinstance(v, MemorySource):
        return v
    s = str(v or "").lower()
    try:
        return MemorySource(s)
    except Exception:
        return MemorySource.USER
