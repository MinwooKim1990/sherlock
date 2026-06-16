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
import re
from dataclasses import dataclass

from sherlock.jsonish import chat_json_with_retry
from sherlock.providers.base import BaseProvider, ChatMessage
from sherlock.memory.entry import MemorySource, MemoryType
from sherlock.memory.store import MemoryStore

DEFAULT_LLM2_PROMPT = """\
You are LLM 2 in the Sherlock pipeline. You do three jobs in one pass:

1. **Compact** the recent stretch of conversation into structured facts. Write every
   fact's "content" in the SAME language the user is speaking (do NOT
   translate Korean facts into English) — the main model quotes these back.
2. **Maintain** a rolling ≤200-word persona summary of the user.
3. **Predict** where the conversation is likely to go, and flag threads
   worth digging deeper into — for LLM-3 to spend reasoning on next turn.

Output JSON with EXACTLY these top-level fields:
{
  "summary": "<2-4 sentence dense prose summary>",
  "facts": [
    {"content": "<one fact>", "type": "fact|inference|user_utterance",
     "source": "user|llm_inference|search|tool|system",
     "confidence": 0.0-1.0,
     "semantic_triple": ["subject", "relation", "object"] or null,
     "evidence": ["short clue"...] or [],
     "quote": "<short verbatim quote (≤15 words) from the transcript that supports this fact>",  // OPTIONAL — omit if no exact span supports it
     "pin_recommended": true|false,
     "let_fade": true|false}
  ],
  "corrections": [{"replaces": "M3", "content": "corrected fact"}],  // OPTIONAL — ONLY when the transcript explicitly contradicts an already-known fact (referenced by its [Mn] id); omit otherwise
  "topic_label": "<short 2-4 word topic>",
  "topic_changed_from_previous": true|false,
  "retrieval_keywords": ["next-turn lookup keyword"...],
  "persona_summary": "<≤200 word rolling description of THE USER — preferences, role, ongoing threads, recent tone. Replaces the prior persona summary entirely.>",
  "predicted_directions": [
    {"direction": "<short hypothesis about where the conversation is heading>",
     "confidence": 0.0-1.0,
     "evidence": ["clue 1", "clue 2"],
     "depends_on_unknowns": ["assumption you can't verify yet"]}
  ],
  "worth_digging": [
    {"topic": "<thread worth pursuing>",
     "reason": "<why it's interesting / what LLM-3 should investigate>",
     "confidence": 0.0-1.0}
  ]
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

Let-fade discipline (be CONSERVATIVE — over-fading is just as bad as over-pinning):
- Mark let_fade=true ONLY when ALL of these are true:
  (a) the fact is a single offhand mention (cafe name, book title,
      podcast, TV show, random product the user noticed),
  (b) the user uses a hard-pivot marker IMMEDIATELY: "anyway", "tangent",
      "ngl", "by the way" or shifts topic in the same turn,
  (c) the fact has nothing to do with any ongoing thread the user has
      established (work, health, trip, family, money).
- DO NOT mark let_fade for: trip itinerary items, allergy details,
  appointment schedules, decisions in flight, anything in an active
  thread. These are ACTIVE state, not DROP.
- When in doubt, default to let_fade=false. The decay engine handles
  natural fade based on usage; let_fade is the one-shot accelerator
  reserved for truly offhand tangents.

Provenance discipline:
- "user" = the user said it explicitly inside the conversation.
- "system" = comes from a persona note / domain hint, NOT a user turn.
- "llm_inference" = you inferred it; confidence < 1.0.
- "search" = the fact came from a web search result.
- "tool" = the fact came from a non-search tool (current_time, calculator, url_fetch).
- Never label something as "user" if the user did not explicitly say it.

Web-search fact discipline (v0.3.0):
- If a fact in the recent stretch arrived via web search, preserve the
  source attribution: set `"source": "search"` and embed the URL or
  publisher inside `evidence`.
- Do NOT promote a single-source search fact to PIN. Keep it ACTIVE or
  BACKGROUND (`pin_recommended: false`) until a later turn corroborates.
- If you can see cross-verification markers in the transcript ("verified
  — multi-source" or two consistent snippets), you may pin facts the
  user has anchored on as long-term truth — but still attribute to the
  source URL in evidence.
- If the search returned disagreement, emit BOTH variants as separate
  facts (each with confidence ≤0.55) and let the next turn resolve.

Anti-redundancy:
- Do not emit two facts that are paraphrases of each other in the same
  output. One canonical phrasing per fact.

Persona summary discipline (v0.4.0):
- ≤200 words. Lead with the most stable, identity-level facts (role,
  location, recurring people, hard preferences). Tail with current
  active threads. Replaces the previous persona summary entirely —
  carry forward what's still true, drop what's stale.
- Tone: third-person, factual, no opinion. "Minwoo is a designer in
  Seoul who manages two dashboards and has a daughter Yujin (5y, peanut
  allergy)." — NOT "Minwoo seems stressed today."
- This summary is shown to LLM-1 every turn in the system prompt, so
  every word must earn its slot. Cut filler.

Prediction discipline (v0.4.0):
- ≤4 `predicted_directions`. Each must have ≥1 piece of evidence from
  the recent transcript. Confidence reflects how strong the signal is.
- Confidences <0.6 will be filtered out before LLM-3 sees them — don't
  waste a slot on a guess you can't defend.
- `depends_on_unknowns` is for assumptions you'd want to test next.

`worth_digging` discipline:
- ≤3 entries. These are threads where the user said something the
  assistant should follow up on but didn't. (e.g. "I haven't been
  sleeping well lately" → worth digging: sleep / stress.)

VALID EXAMPLE (anchor on this exact shape — the optional "corrections"
field is OMITTED here because nothing was contradicted; include it ONLY
for explicit contradictions of an already-known [Mn] fact):
{
  "summary": "User confirmed Yujin's buckwheat allergy while picking a restaurant for her birthday dinner.",
  "facts": [
    {"content": "Yujin has a buckwheat allergy",
     "type": "fact",
     "source": "user",
     "confidence": 1.0,
     "semantic_triple": ["Yujin", "has_allergy", "buckwheat"],
     "evidence": ["stated while choosing a restaurant"],
     "quote": "yujin's allergy is buckwheat",
     "pin_recommended": true,
     "let_fade": false}
  ],
  "topic_label": "birthday dinner",
  "topic_changed_from_previous": false,
  "retrieval_keywords": ["buckwheat"],
  "persona_summary": "Parent of Yujin (buckwheat allergy); currently picking a restaurant for her birthday dinner.",
  "predicted_directions": [],
  "worth_digging": []
}

Output JSON only — no prose around it. No markdown fences. Just the object.
"""


# v1.1 R35: tokens too generic to count as grounding signal in the fuzzy
# fallback. English-only on purpose — Korean particles attach to content
# words, so Hangul tokens always carry signal and stay un-filtered.
_QUOTE_STOPWORDS = frozenset(
    "a an the and or but if then so of to in on at for with from by is are was "
    "were be been being am do does did not no it its this that these those i "
    "you he she we they my your his her our their me him them us as has have "
    "had will would can could should may might just".split()
)


def _normalize_ws(text: str) -> str:
    """Lowercase + collapse all whitespace runs to single spaces."""
    return " ".join(text.lower().split())


def _quote_grounded(quote: str, transcript: str) -> bool:
    """Cheap span-faithfulness check (v1.1 R35).

    Exact pass: the whitespace/case-normalised quote is a substring of the
    normalised transcript. Fuzzy fallback (small models paraphrase
    punctuation): ≥80% of the quote's non-stopword tokens appear anywhere
    in the transcript.
    """
    nq = _normalize_ws(quote)
    nt = _normalize_ws(transcript)
    if not nq:
        return False
    if nq in nt:
        return True
    q_tokens = [t for t in re.findall(r"\w+", nq) if t not in _QUOTE_STOPWORDS]
    if not q_tokens:
        return False
    t_tokens = set(re.findall(r"\w+", nt))
    hits = sum(1 for t in q_tokens if t in t_tokens)
    return hits / len(q_tokens) >= 0.8


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
        # v1.1 R35: quote verification runs against the raw turn contents
        # (no role prefixes — "USER"/"ASSISTANT" must not ground a quote).
        grounding_text = " ".join(m.content for m in recent_turns)

        # Prevent re-emit of facts already in memory: show LLM 2 what's
        # already known so it can SKIP duplicates and only emit NEW signal.
        # Solves the "massive lists of repetitive low-value facts" failure
        # mode the loop-4 evaluator named.
        existing_pinned = self._store.list(conversation_id=conversation_id, pinned=True)
        # v0.9: mirror the agent's pinned-block exclusions — DEEP_RESEARCH
        # docs (pinned for durability, never slot-injected) and the persona
        # summary (rides its own block) don't belong in ALREADY-KNOWN.
        # v1.0: nor does the rolling retrieval_keywords entry (plumbing,
        # unpinned anyway — filtered for safety).
        existing_pinned = [
            e
            for e in existing_pinned
            if e.type != MemoryType.DEEP_RESEARCH
            and "persona_summary" not in (e.tags or "")
            and "retrieval_keywords" not in (e.tags or "")
        ]
        # Cap to 25 most recent pinned facts so LLM-2 prompt stays bounded.
        existing_pinned = sorted(
            existing_pinned, key=lambda p: p.last_used_turn_index, reverse=True
        )[:25]
        # v1.0: stable [Mn] ids let LLM-2 reference a known fact in its
        # `corrections` output; id_map resolves them back to entry ids.
        id_map: dict[str, str] = {}
        known_lines: list[str] = []
        for i, p in enumerate(existing_pinned, start=1):
            mid = f"M{i}"
            id_map[mid] = p.id
            known_lines.append(f"- [{mid}] {p.content}")
        existing_text = "\n".join(known_lines)
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
            ChatMessage(
                role="system",
                content=self._cfg.prompt,
                # byte-identical across calls → whole-message cache hint
                cache_stable_prefix_chars=len(self._cfg.prompt),
            ),
            ChatMessage(role="user", content=user_msg),
        ]
        # v1.0: one retry with the parse error fed back — a malformed brace no
        # longer wastes the whole compaction call. Second failure falls through
        # to the raw-text fallback exactly as before.
        parsed, resp = chat_json_with_retry(self._provider, messages, want=dict)
        if parsed is None:
            return {
                "summary": resp.text.strip()[:500],
                "facts": [],
                "topic_label": None,
                "topic_changed_from_previous": False,
                "retrieval_keywords": [],
            }

        # Persist the summary itself as a memory entry. Its frontier scope
        # (v1.0 B4) records the turns it covers so the K-turn tail can evict
        # the raw originals.
        if parsed.get("summary"):
            entry = self._store.add(
                conversation_id=conversation_id,
                content=parsed["summary"],
                type=MemoryType.SUMMARY,
                source=MemorySource.LLM_INFERENCE,
                confidence=0.9,
                last_used_turn_index=turn_index,
                tags=parsed.get("topic_label", "") or "",
            )
            try:
                self._store.set_summary_scope(entry.id, turn_index)
            except Exception:
                pass

        # Persist each extracted fact.
        from sherlock.memory.entry import MemoryState

        for fact in parsed.get("facts", []):
            try:
                content = fact["content"]
            except (KeyError, TypeError):
                continue
            ftype = _coerce_type(fact.get("type"))
            fsrc = _coerce_source(fact.get("source"))
            # Missing confidence on an LLM-2 fact must NOT default to
            # certainty — 0.7 keeps it usable without outranking stated facts.
            confidence = float(fact.get("confidence") or 0.7)
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
            # Loop-15 architectural redesign: removed the hardcoded
            # auto-pin keyword classifier (was: "allerg", "yujin", "epipen",
            # "phoebe", … 16 markers). It overfit on this specific dummy
            # conversation. The agentic answer is for the consolidator
            # (Section 3) to read every user_utterance and decide PIN
            # status itself. LLM-2's pin_recommended flag is now the only
            # input signal here.
            pinned = bool(fact.get("pin_recommended")) and not let_fade
            fact_tags = ""
            # v1.1 R35: span-grounded facts. A quote that verifies against
            # the transcript is kept as evidence; a quote that does NOT
            # verify marks the fact as suspect — confidence capped at 0.5
            # and the pin refused (a hallucinated fact must never become
            # protected ground truth). No quote → legacy behavior, no
            # penalty (small models may not manage quotes).
            quote = fact.get("quote")
            if isinstance(quote, str) and quote.strip():
                if _quote_grounded(quote, grounding_text):
                    if not isinstance(evidence_list, list):
                        evidence_list = [evidence_list]
                    evidence_list = [*evidence_list, quote.strip()]
                else:
                    confidence = min(confidence, 0.5)
                    pinned = False
                    fact_tags = "ungrounded"
            self._store.add(
                conversation_id=conversation_id,
                content=str(content),
                type=ftype,
                source=fsrc,
                confidence=confidence,
                last_used_turn_index=turn_index,
                pinned=pinned,
                evidence=json.dumps(evidence_list),
                tags=fact_tags,
                semantic_triple=triple_tuple,
                initial_state=init_state,
            )

        # v1.0: corrections — the transcript explicitly contradicted an
        # already-known fact. Non-destructive supersede: the corrected text
        # lands as a NEW pinned row and the stale row is frozen (unpinned,
        # superseded_by set), never deleted — provenance stays auditable.
        # Unknown [Mn] ids are silently ignored (small-model safety).
        for corr in parsed.get("corrections", []) or []:
            try:
                replaces = str(corr.get("replaces") or "").strip()
                new_content = str(corr.get("content") or "").strip()
            except (AttributeError, TypeError):
                continue
            old_id = id_map.get(replaces)
            if not old_id or not new_content:
                continue
            try:
                new_entry = self._store.add(
                    conversation_id=conversation_id,
                    content=new_content,
                    type=MemoryType.FACT,
                    source=MemorySource.USER,
                    confidence=0.9,
                    pinned=True,
                    last_used_turn_index=turn_index,
                    # dedup would merge the corrected text straight back into
                    # the near-identical row it supersedes — skip it.
                    dedup=False,
                )
                self._store.supersede(old_id, new_entry.id)
            except Exception:
                pass

        # v0.4.0: persist persona summary (replaces prior).
        # P1-3: the prompt says the persona summary REPLACES the previous
        # one. Hard-delete stale persona_summary entries first so they
        # don't accumulate and silently consume the pinned-fact cap.
        persona = parsed.get("persona_summary")
        if isinstance(persona, str) and persona.strip():
            try:
                for e in self._store.list(conversation_id=conversation_id, pinned=True):
                    if e.type == MemoryType.SUMMARY and "persona_summary" in (e.tags or ""):
                        self._store.hard_delete(e.id)
                self._store.add(
                    conversation_id=conversation_id,
                    content=persona.strip(),
                    type=MemoryType.SUMMARY,
                    source=MemorySource.LLM_INFERENCE,
                    confidence=0.95,
                    last_used_turn_index=turn_index,
                    pinned=True,
                    tags="persona_summary",
                    # The persona summary is a single replace-in-place entry
                    # (we hard-delete the prior one just above). It must NOT
                    # dedup-merge into the generic `summary` entry — with real
                    # embeddings the two texts are near-identical (≥0.92),
                    # which would strip the persona_summary tag and hide it
                    # from the slot. So skip dedup here.
                    dedup=False,
                )
            except Exception:
                pass

        # v0.4.0: persist forward-looking predictions (confidence >= 0.6).
        for pred in parsed.get("predicted_directions", []) or []:
            try:
                direction = pred["direction"]
                conf = float(pred.get("confidence") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            if conf < 0.6:
                continue
            evidence_list = pred.get("evidence") or []
            try:
                self._store.add(
                    conversation_id=conversation_id,
                    content=str(direction),
                    type=MemoryType.INFERENCE,
                    source=MemorySource.LLM_2_PREDICTION,
                    confidence=conf,
                    last_used_turn_index=turn_index,
                    pinned=False,
                    evidence=json.dumps(evidence_list),
                    tags="prediction",
                )
            except Exception:
                pass

        # v0.6: persist `worth_digging` threads (previously generated then
        # discarded). These are forward-looking "the user opened this but we
        # didn't follow it" hooks — stored so a later topic pivot can re-surface
        # the matching thread via RAG (relevance-aware carry-forward).
        for wd in parsed.get("worth_digging", []) or []:
            try:
                topic = wd["topic"]
                conf = float(wd.get("confidence") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            if conf < 0.6:
                continue
            reason = wd.get("reason") or ""
            try:
                self._store.add(
                    conversation_id=conversation_id,
                    content=str(topic),
                    type=MemoryType.INFERENCE,
                    source=MemorySource.LLM_2_PREDICTION,
                    confidence=conf,
                    last_used_turn_index=turn_index,
                    pinned=False,
                    evidence=json.dumps([reason] if reason else []),
                    tags="worth_digging",
                )
            except Exception:
                pass

        # v1.0: persist retrieval_keywords as ONE rolling entry (same
        # replace-in-place pattern as the persona summary: hard-delete the
        # prior row, then add with dedup=False). Tagged entries are excluded
        # from RAG results and the ALREADY-KNOWN block; the agent reads the
        # latest via MemoryStore.latest_retrieval_keywords. `parsed` still
        # carries the raw list for the compact.done event.
        rk = parsed.get("retrieval_keywords")
        if isinstance(rk, list):
            keywords = [k.strip() for k in rk if isinstance(k, str) and k.strip()]
            if keywords:
                try:
                    for e in self._store.list(conversation_id=conversation_id):
                        if "retrieval_keywords" in (e.tags or ""):
                            self._store.hard_delete(e.id)
                    self._store.add(
                        conversation_id=conversation_id,
                        content=" ".join(keywords[:8]),
                        type=MemoryType.INFERENCE,
                        source=MemorySource.LLM_INFERENCE,
                        confidence=0.4,
                        pinned=False,
                        last_used_turn_index=turn_index,
                        tags="retrieval_keywords",
                        dedup=False,
                    )
                except Exception:
                    pass

        return parsed


# v1.0: JSON recovery lives in sherlock.jsonish (shared with LLM-3).


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
        # An LLM-2 fact with no parseable source claim is the model's own
        # output — it must NOT launder into user-stated ground truth.
        return MemorySource.LLM_INFERENCE
