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

Pinning discipline (this is where most systems fail):
- Pin ONLY facts the user clearly wants permanently remembered: their
  location, their role, the names of family members, allergies, key
  dates, hard preferences, contracts. Default pin=false.
- Do NOT pin transient state (current task progress, current decision-
  in-flight, "I'm tired right now"). That is ACTIVE state, which lives
  unpinned and decays naturally.
- Do NOT re-pin the same fact every turn. If a fact is already in
  memory, skip it instead of re-emitting an identical entry.

Let-fade discipline (the counterpart):
- Mark let_fade=true for offhand mentions immediately followed by an
  "anyway" / "tangent" / hard pivot to the next topic. Cafes, books,
  podcasts, random TV shows, one-off observations.
- A single mention with no return is a fade signal. The system will
  decay these automatically — but you must mark them so it knows.

Provenance discipline:
- "user" = the user said it explicitly inside the conversation.
- "system" = comes from a persona note / domain hint, NOT a user turn.
- "llm_inference" = you inferred it; confidence < 1.0.
- Never label something as "user" if the user did not explicitly say it.

Anti-redundancy:
- Do not emit two facts that are paraphrases of each other in the same
  output. One canonical phrasing per fact.

Output JSON only — no prose around it. No markdown fences. Just the object.
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

        # Prevent re-emit of facts already in memory: show LLM 2 what's
        # already known so it can SKIP duplicates and only emit NEW signal.
        # Solves the "massive lists of repetitive low-value facts" failure
        # mode the loop-4 evaluator named.
        existing_pinned = self._store.list(conversation_id=conversation_id, pinned=True)
        # Cap to 25 most recent pinned facts so LLM-2 prompt stays bounded.
        existing_pinned = sorted(
            existing_pinned, key=lambda p: p.last_used_turn_index, reverse=True
        )[:25]
        existing_text = "\n".join(f"- {p.content}" for p in existing_pinned)
        if not existing_text:
            existing_text = "(none yet)"

        user_msg = (
            "Here is the most recent stretch of conversation. Produce the "
            "JSON described in your system prompt.\n\n"
            "--- ALREADY-KNOWN FACTS (do NOT re-emit these; only emit NEW signal) ---\n"
            f"{existing_text}\n"
            "--- END ALREADY-KNOWN ---\n\n"
            "--- TRANSCRIPT ---\n"
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
        from sherlock.memory.entry import MemoryState

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
            # let_fade=true means LLM-2 thinks this is offhand. Land it
            # directly in COLD so it's RAG-retrievable but not in the slot,
            # and the next decay pass will move it to FORGOTTEN if it
            # remains unreferenced.
            let_fade = bool(fact.get("let_fade"))
            init_state = MemoryState.COLD if let_fade else MemoryState.FRESH
            self._store.add(
                conversation_id=conversation_id,
                content=str(content),
                type=ftype,
                source=fsrc,
                confidence=confidence,
                last_used_turn_index=turn_index,
                pinned=bool(fact.get("pin_recommended")) and not let_fade,
                evidence=json.dumps(evidence_list),
                semantic_triple=triple_tuple,
                initial_state=init_state,
            )

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
