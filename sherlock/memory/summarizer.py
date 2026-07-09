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
from sherlock.memory.entry import (
    LTM_CONVERSATION_ID,
    MemoryEntry,
    MemorySource,
    MemoryType,
    ltm_category,
)
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
- Tone: third-person, factual, no opinion. "Alex is a designer in
  Berlin who manages two dashboards and has a daughter (5y, peanut
  allergy)." — NOT "Alex seems stressed today."
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


# v1.12 Stage A1: ADDITIVE instruction block, concatenated onto the LLM-2
# system prompt at run-time ONLY when long-term memory is enabled (and not
# incognito). Off → the prompt is byte-identical to DEFAULT_LLM2_PROMPT, so we
# never edit the string above. Teaches two extra per-fact fields; a code-level
# gate (never the model alone) still decides what is actually promoted.
LTM_PROMPT_SUFFIX = """\

LONG-TERM MEMORY (cross-conversation) — extra per-fact fields:
For EACH fact, ALSO decide whether it should be remembered PERMANENTLY, across
future conversations (not just this session). Add two fields to each fact:
  "long_term": true|false,
  "category": "user_directive|identity_health|stable_preference|relationship|long_term_project|none"

Category taxonomy (be strict — durable memory is expensive to get wrong):
- user_directive — the user EXPLICITLY asked you to remember it ("remember
  that…", "from now on…", "always…"). ALWAYS long_term.
- identity_health — the user's name, pronouns, allergies, medical conditions,
  or other stable identity/health facts. ALWAYS long_term.
- stable_preference — a durable, repeated preference (not a one-off mood).
- relationship — a stable relationship (family member, colleague, pet) and who
  they are to the user.
- long_term_project — an ongoing multi-session project, goal, or commitment.
- none — transient tasks, in-flight decisions, one-off mentions, speculation,
  or anything you're unsure about. Set long_term=false. This is the DEFAULT.

Only set long_term=true for stable_preference / relationship / long_term_project
when the fact is CLEARLY durable. When in doubt, category="none",
long_term=false. Keep every "content" in the user's own language (never
translate). These fields are ADVISORY — never restate a fact just to promote it.
"""

# Code-level promotion gate (never trust the model's booleans alone).
_LTM_ALWAYS_CATEGORIES = frozenset({"user_directive", "identity_health"})
_LTM_CONSERVATIVE_CATEGORIES = frozenset({"stable_preference", "relationship", "long_term_project"})
_LTM_CONSERVATIVE_MIN_CONFIDENCE = 0.7


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
        long_term=None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._cfg = config or SummarizerConfig()
        # v1.12 Stage A1: LongTermMemoryConfig (or None → feature off). Held by
        # reference so the agent/playground can flip `.enabled` live. None and
        # ``enabled=False`` both mean: prompt byte-identical, zero LTM writes.
        self._long_term = long_term

    def _ltm_active(self) -> bool:
        """Long-term promotion writes enabled this run (enabled AND not incognito)."""
        lt = self._long_term
        return bool(lt and getattr(lt, "enabled", False) and not getattr(lt, "incognito", False))

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
        promote_user_directive: bool = False,
    ) -> dict:
        """Call LLM 2 over the recent-turn window. Returns the parsed JSON dict.

        v1.12 Stage A3: when ``promote_user_directive`` is set (a deterministic
        "remember this" cue fired on a covered turn), every fact in this window is
        promoted to long-term memory under the ALWAYS category ``user_directive``
        — belt-and-braces behind LLM-1's explicit ``memory save``. Inert when
        long-term promotion is not active (feature off / incognito)."""
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

        # v1.12 Stage A1: append the long-term instruction block ONLY when the
        # feature is active (enabled AND not incognito). Off → byte-identical to
        # the base prompt, so the whole-message cache hint is unchanged too.
        system_prompt = self._cfg.prompt
        if self._ltm_active():
            system_prompt = self._cfg.prompt + LTM_PROMPT_SUFFIX
        messages = [
            ChatMessage(
                role="system",
                content=system_prompt,
                # byte-identical across calls → whole-message cache hint
                cache_stable_prefix_chars=len(system_prompt),
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
                # v1.12 Stage A1: always present so the agent can emit uniformly.
                "long_term_promoted": [],
            }

        # v1.12 Stage A1: accumulate long-term promotions for this run (facts +
        # correction propagation). Surfaced in the result dict; empty when the
        # feature is off. ``_ltm_active`` is read once — a stable gate per run.
        long_term_promoted: list[dict] = []
        ltm_active = self._ltm_active()

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
            quote_text = quote.strip() if isinstance(quote, str) and quote.strip() else ""
            quote_is_grounded = False
            if quote_text:
                if _quote_grounded(quote_text, grounding_text):
                    quote_is_grounded = True
                    if not isinstance(evidence_list, list):
                        evidence_list = [evidence_list]
                    evidence_list = [*evidence_list, quote_text]
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
            # v1.12 Stage A1: cross-conversation LONG-TERM promotion. The model
            # tags each fact with a category; the CODE gate below decides — the
            # model's flag alone never promotes anything.
            if ltm_active:
                # v1.12 A3: a latched "remember this" cue forces every covered
                # fact to the ALWAYS user_directive category (flag-independent),
                # so the durable intent survives even if LLM-1 skipped `memory
                # save`. The ungrounded-quote guard in _maybe_promote_long_term
                # still blocks a hallucinated-quote fact from being made durable.
                if promote_user_directive:
                    fact = {**fact, "category": "user_directive", "long_term": True}
                self._maybe_promote_long_term(
                    fact=fact,
                    conversation_id=conversation_id,
                    content=str(content),
                    source=fsrc,
                    confidence=confidence,
                    turn_index=turn_index,
                    quote_text=quote_text,
                    quote_is_grounded=quote_is_grounded,
                    promoted_out=long_term_promoted,
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
                # Capture the stale text BEFORE supersede so a long-term copy
                # can be matched by content hash (supersede leaves it intact).
                old_row = self._store.get(old_id)
                old_content = old_row.content if old_row is not None else None
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
                # v1.1 R34 bug fix: pass the turn index so the frozen row's
                # ``invalid_at_turn`` is populated (bi-temporal validity). run()
                # already receives turn_index; the correction path just never
                # forwarded it, so "what was true before turn X?" was unanswerable.
                self._store.supersede(old_id, new_entry.id, turn_index=turn_index)
                # v1.12 Stage A1: propagate the correction to any long-term copy
                # of the same fact (matched by content hash) so durable memory
                # doesn't keep asserting the stale value.
                if ltm_active and old_content:
                    self._supersede_long_term(
                        old_content=old_content,
                        new_content=new_content,
                        conversation_id=conversation_id,
                        turn_index=turn_index,
                        promoted_out=long_term_promoted,
                    )
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

        # v1.12 Stage A1: bound the long-term store. Past the cap, drop the
        # lowest-confidence / oldest promoted rows (best-effort). Only runs
        # when something was promoted this cycle, so the off/no-op path is free.
        if ltm_active and long_term_promoted:
            self._enforce_ltm_cap()

        # v1.12 Stage A1: surface promotions for the agent's memory.promoted
        # event. Always present (empty when the feature is off or nothing durable).
        parsed["long_term_promoted"] = long_term_promoted

        return parsed

    # ---------------- v1.12 Stage A1: long-term promotion helpers -----------

    def _maybe_promote_long_term(
        self,
        *,
        fact: dict,
        conversation_id: str,
        content: str,
        source: MemorySource,
        confidence: float,
        turn_index: int,
        quote_text: str,
        quote_is_grounded: bool,
        promoted_out: list[dict],
    ) -> None:
        """Code-level promotion gate — never trusts the model's flags alone.

        ALWAYS categories (user_directive / identity_health) promote. CONSERVATIVE
        categories (stable_preference / relationship / long_term_project) require
        confidence ≥ 0.7 AND a grounded quote. Anything else is skipped.
        """
        try:
            category = str(fact.get("category") or "none").strip().lower()
            # v1.12 F1: a quote that FAILED grounding (v1.1 R35 marked the fact
            # suspect — confidence capped 0.5, pin refused, tag "ungrounded")
            # must never become permanent pinned memory, even under an ALWAYS
            # category. No quote → nothing was grounded to fail, so the ALWAYS
            # "no quote still promotes" contract is preserved.
            if quote_text and not quote_is_grounded:
                return
            if category in _LTM_ALWAYS_CATEGORIES:
                pass  # always durable (flag-independent per contract)
            elif category in _LTM_CONSERVATIVE_CATEGORIES:
                # v1.12 F3: a durable category alone over-promotes — also require
                # the model's explicit long_term=True flag for conservative rows
                # (ALWAYS categories stay flag-independent per contract).
                if (
                    confidence < _LTM_CONSERVATIVE_MIN_CONFIDENCE
                    or not quote_is_grounded
                    or fact.get("long_term") is not True
                ):
                    return
            else:
                return  # category "none"/unknown → never promote
            # Origin turn + source quote ride in the evidence JSON list.
            evidence = [{"quote": quote_text, "turn": turn_index}]
            row = self._store.add(
                conversation_id=LTM_CONVERSATION_ID,
                content=content,
                type=MemoryType.FACT,
                source=source,
                confidence=confidence,
                pinned=True,
                last_used_turn_index=turn_index,
                tags=f"ltm,{category}",
                evidence=json.dumps(evidence),
                origin_conversation_id=conversation_id,
                # Sentinel-scoped dedup gives cross-conversation merge for free.
                # v1.12 F8 (accepted limitation): store's prefix-dedup pass
                # (store.py ~214-257) treats a 60-char shared prefix as the same
                # fact, so two DISTINCT long-term facts that happen to share a
                # long prefix can collapse — the survivor's content is rewritten
                # in place but its tags/origin_conversation_id are NOT updated,
                # and the "recent 40" ordering window is incoherent here since
                # every sentinel row shares one conversation_id. Tolerated for
                # Stage A1; a sentinel-aware dedup key is a later-stage fix.
                dedup=True,
            )
            promoted_out.append({"content": content, "category": category, "id": row.id})
        except Exception:
            pass

    def _supersede_long_term(
        self,
        *,
        old_content: str,
        new_content: str,
        conversation_id: str,
        turn_index: int,
        promoted_out: list[dict],
    ) -> None:
        """Propagate a session correction to any long-term copy of the same fact.

        Promote the corrected text as a fresh long-term row and non-destructively
        supersede the stale one (carrying its category + populating
        invalid_at_turn).

        v1.12 F4 (limitation): the stale sentinel is located by EXACT content
        hash, with a conservative 60-char-prefix fallback (mirroring store's
        prefix-dedup) when the hash misses. The hash can legitimately miss when
        promotion dedup-merged this fact semantically (a ≥0.92 match keeps the
        EXISTING sentinel content, not the new text) or via a prefix correction;
        if such a merge also diverges within the first 60 chars, the correction
        cannot be located and durable memory keeps the stale value. This is not
        guaranteed to share a hash — the older docstring overstated that."""
        try:
            old_hash = MemoryEntry.compute_hash(old_content)
            old_norm = " ".join((old_content or "").strip().lower().split())
            live_rows = [
                r
                for r in self._store.list(conversation_id=LTM_CONVERSATION_ID)
                if r.superseded_by is None
            ]
            matches = [r for r in live_rows if r.content_hash == old_hash]
            if not matches and len(old_norm) > 30:
                # Hash missed — fall back to store's conservative prefix shape
                # (len>30, identical 60-char prefix) so a dedup-merged sentinel
                # still gets corrected.
                for r in live_rows:
                    rnorm = " ".join((r.content or "").strip().lower().split())
                    if len(rnorm) > 30 and rnorm[:60] == old_norm[:60]:
                        matches.append(r)
            for ltm_row in matches:
                category = ltm_category(ltm_row.tags)
                new_row = self._store.add(
                    conversation_id=LTM_CONVERSATION_ID,
                    content=new_content,
                    type=MemoryType.FACT,
                    source=MemorySource.USER,
                    confidence=0.9,
                    pinned=True,
                    last_used_turn_index=turn_index,
                    tags=ltm_row.tags or f"ltm,{category}",
                    evidence=json.dumps([{"quote": "", "turn": turn_index}]),
                    origin_conversation_id=conversation_id,
                    # A correction must not dedup-merge back into the row it replaces.
                    dedup=False,
                )
                self._store.supersede(ltm_row.id, new_row.id, turn_index=turn_index)
                promoted_out.append(
                    {"content": new_content, "category": category, "id": new_row.id}
                )
        except Exception:
            pass

    def _enforce_ltm_cap(self) -> None:
        """Hard-delete the lowest-confidence / oldest long-term rows past the cap.

        Live rows evict by (is-ALWAYS-category, confidence, created_at) so a
        just-promoted user_directive/identity_health outlives a higher-confidence
        conservative row (v1.12 F2). Superseded "frozen" rows — one added per
        correction and previously exempt from the cap forever — are also bounded
        oldest-first so repeated corrections can't grow the store unbounded
        (v1.12 F5)."""
        try:
            cap = int(getattr(self._long_term, "cap", 200) or 200)
            all_rows = self._store.list(conversation_id=LTM_CONVERSATION_ID)
            live = [e for e in all_rows if e.superseded_by is None]
            if len(live) > cap:
                # v1.12 F2: ALWAYS categories evaluate to True and sort LAST, so
                # conservative rows are evicted before a durable directive/identity
                # fact. Within a category: lowest confidence, then oldest, go first.
                live.sort(
                    key=lambda e: (
                        ltm_category(e.tags) in _LTM_ALWAYS_CATEGORIES,
                        e.confidence,
                        e.created_at,
                    )
                )
                for e in live[: len(live) - cap]:
                    self._store.hard_delete(e.id)
            # v1.12 F5: frozen (superseded) sentinel rows are otherwise immortal
            # — bound them oldest-first so a long correction history stays capped.
            frozen = [e for e in all_rows if e.superseded_by is not None]
            if len(frozen) > cap:
                frozen.sort(key=lambda e: e.created_at)
                for e in frozen[: len(frozen) - cap]:
                    self._store.hard_delete(e.id)
        except Exception:
            pass


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
