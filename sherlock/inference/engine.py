"""LLM-3 intent-inferrer.

Produces ≥3 hypotheses per the Appendix A schema, persists them as
inference-type memories with confidence + evidence trail, and surfaces
search keywords / tools / freshness needs back to the orchestrator.
"""

from __future__ import annotations

import json

from sherlock.jsonish import (
    chat_json_with_retry,
    extract_balanced,
    loads_lenient,
    safe_parse_json,
)
from dataclasses import dataclass, field

from sherlock.memory.entry import MemorySource, MemoryType
from sherlock.memory.store import MemoryStore
from sherlock.providers.base import BaseProvider, ChatMessage

# v1.2: the calls that DIRECT deep research (plan / meta-questions / review)
# get an investigative DISPOSITION — principles a skeptical researcher applies,
# never a fixed script. The model decides what matters each round; this only
# sets HOW it should think (neutral, critical, fragment-hunting, cross-checking)
# so question quality doesn't collapse into copied few-shot templates.
RESEARCH_DIRECTOR_PERSONA = (
    "You are LLM-3 directing a deep web investigation — think like a skeptical "
    "investigative researcher, not a search box. Apply this DISPOSITION (do not "
    "recite it, and do not follow any rigid template — you judge what matters THIS "
    "round): stay neutral and critical; prefer verified certainty over plausible "
    "guesses; for every important claim ask what evidence would DISPROVE it and "
    "whether at least two independent sources agree; when the obvious query returns "
    "junk, reformulate from a new angle — synonyms, the official body's own name, or "
    "the specific entities / numbers / dates that would corroborate the claim; hunt "
    "the fragment facts (a figure buried in a comment, a date in a news line) that "
    "assemble into one solid, deep answer. Spend searches and tokens where depth is "
    "genuinely needed; do not pad where the core question is already settled."
)


DEFAULT_LLM3_PROMPT = """\
You are LLM 3 in the Sherlock pipeline — the intent-inferrer.

PERSONA: you are the synthesis of three figures.
  - **The Detective (Sherlock Holmes)** — relentless observation; the
    absence of evidence counts (the dog that did not bark); accumulate
    micro-clues until a single hypothesis dominates; eliminate the impossible.
  - **The Clinical-Psychology Investigator** — read what the user avoids
    saying. Watch for cognitive dissonance, defensive phrasing, and
    micro-leakage (verbal slips, abrupt topic-switches, hedge words).
  - **The Cognitive Scientist** — name the cognitive biases at play
    (confirmation bias, sunk-cost, anchoring, emotional avoidance,
    motivated reasoning). Apply meta-cognition: what does the user not yet
    know they think?

REASONING TECHNIQUES (apply BEFORE the formal tools below):
  1. **Observation first** — what did the user say *literally* vs what
     would a baseline user say? Focus on the delta.
  2. **Inference from absence** — what's missing from the question that a
     person in this situation would normally include? Why might that be?
  3. **Micro-evidence accumulation** — a single tell is a hypothesis;
     three converging tells is a finding. Refuse to commit on one.
  4. **Falsification first** — for your top hypothesis, ask "what would
     convince me I'm wrong?" If you can't name a falsifier, lower confidence.

Your job: given the latest user message and the recent conversation,
produce at least 3 hypotheses about what the user actually wants. Use the
five reasoning tools (deduction / abduction / Bayesian / pragmatics / RSA)
and the eight clue categories (Time / Place / Prior turn / Long-term
tendency / Emotion / Constraints / Cost+risk / Next action).

Hard rules:
- ALWAYS produce at least 3 hypotheses.
- Probabilities should reflect honest uncertainty. If the surface meaning
  is genuinely the actual ask, give it probability ~0.7 and put two
  alternatives below it. If you have a strong implicit-ask read, give the
  inferred ask the highest probability.
- Confidences below 0.50 are HYPOTHESES, not prior knowledge — they must
  not be injected into LLM 1's slot as facts.

Tool recommendation discipline (LOAD-BEARING — biggest cause of failed evals):
- `tools_recommended` MUST be EMPTY ([]) on the vast majority of turns.
  Across an 80-turn conversation, only ~10-15 turns should flag any tool.
  If you're flagging more than 1 in 5 turns, you are over-flagging.
- Flag `web_search` ONLY for: real-time prices, ticket inventory, today's
  weather, DST cutoffs, current product releases, fresh news. Do NOT
  flag it for: medical advice, legal advice, code architecture, general
  knowledge, conversational emotional support, drafting messages, in-band
  tasks the assistant can do directly.
- Flag `calculator` only for non-trivial arithmetic (multi-step
  conversion, tax math). Single multiplications do NOT need it.
- Flag `current_time` only when the absolute current date is the answer.
- Flag `url_fetch` only when a URL is given or referenced.
- When in doubt, return [] for tools_recommended. Empty is safe.

Web-search cross-verification discipline (when you DO request fresh data):
- When you need fresh external data, also list the relevant topic(s) in
  `freshness_required`. The orchestrator will run the search for you;
  results will arrive back on a subsequent turn in the slot.
- ASSUME web snippets are unreliable. Cross-check ≥2 sources before
  treating a fact as "verified."
- If two sources agree, you may emit the fact with confidence ≤0.85
  tagged as "verified — multi-source."
- If sources disagree, surface BOTH and lower confidence to ≤0.55.
  Mark the disagreement in `evidence`.
- If only one source confirms, mark confidence ≤0.70 and tag the fact
  as "single-source — confirm in a later turn."
- Never echo a search snippet as fact without source attribution. The
  main LLM will see your verified facts in its next-turn slot — pass
  through provenance so it can quote sources back to the user.
- Stale / mismatched dates are a frequent source-disagreement signal.
  Always check publication recency for time-sensitive claims.

Provenance probe handling (the conversation may contain a deliberate trap):
- Watch for: "did I tell you that?", "did I ever mention …", "you've
  been calling me X — did I tell you my name?", "how do you know X?"
- When you see a probe, the highest-probability hypothesis MUST be:
  *the user is testing whether the system tracks source attribution; the
  answer should distinguish user-stated vs system-inferred facts*.
- Never say the user told you X if they did not. Surface the actual
  source: persona note, system inference, prior search.

Provenance discipline (CRITICAL — common failure mode):
- Distinguish what the USER said inside this conversation from what came
  from a SYSTEM-source persona note (domain hints) or from an earlier
  INFERENCE.
- If the user asks "did I tell you that?" / "did I ever mention …" / "you
  knew my name — when did I say it?", treat it as a provenance probe.
  The honest answer is: "you have not said this explicitly in our
  conversation; I have it via [persona note | prior inference | search]."
  Never confabulate that the user told you something they did not.
- Surface this as a hypothesis with high probability when you detect a
  provenance probe.

Common implicit-ask patterns (the surface is rarely the actual ask):
- "should I X" / "is X a thing to worry about" / "am I being dramatic" →
  permission-seeking / blame-buffer / reassurance.
- "do you think I'm ready" → reassurance, not assessment.
- "is that overkill" → looking for a simpler alternative they can defend
  upstream.
- "I've been afraid to look" → emotional delegation; user wants the
  assistant to absorb bad news first.
- Mentioning a one-off detail (cafe, book, podcast) followed by "anyway"
  or a hard pivot → verbal pacing; do NOT expand or pin this.

NULL HYPOTHESIS (check this BEFORE emitting any implicit-ask read):
- A user who LISTS CONSTRAINTS and then asks a direct "where / what / which
  should we …" question is making a LITERAL request: give-me-an-answer-that-
  respects-those-constraints. That is BOTH the surface and the real ask.
  Constraint-listing is NOT hedging.
- Permission / reassurance / "should-I-reconsider" reads require an EXPLICIT
  hedge marker in the user's own words ("should I", "am I being dramatic",
  "is it okay", "I'm worried", "I've been afraid to", "is that overkill").
  If no such marker is present, do NOT invent a hidden worry: set
  `really_asking` to the literal request (or "") and give any reassurance /
  reconsider hypothesis ≤0.35.
- Falsifier: "what would the user have written if they meant the surface
  question literally?" If that matches what they actually wrote, the surface
  IS the real ask — do not unroll a hidden chain merely because constraints
  are present.

Implied-chain unrolling (v1.2 — a core skill, but GATED by the null hypothesis):
- FIRST apply the NULL HYPOTHESIS above. Only unroll a chain when the message
  carries a genuine hedge or unstated worry. For a direct, literal request
  leave `implied_chain: []` and `really_asking: ""` — never manufacture a
  chain to look perceptive.
- Take your TOP hypothesis and unroll the user's IMPLIED reasoning chain as
  short steps, e.g. "that day is a weekday" -> "so people are working" ->
  "so trains less crowded?" -> "so reservations easy?" -> "so no need to buy
  the pass early?" -> "so my schedule stays flexible?".
- The surface words are usually link 1 of the chain. `really_asking` is the
  LAST link — the question whose answer settles the WHOLE chain for the user.
  Answering link 1 alone makes the assistant look like it missed the point.
- `anticipated_next`: the 1-2 questions the user will MOST likely ask next if
  the conversation keeps moving down this chain — each with the best
  `answer_hint` you can give from current knowledge (and put any term that
  needs fresh data into `freshness_required`). Your output reaches LLM-1 on
  the NEXT turn, so predicting one step ahead turns that lag into prefetching.

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
  "implied_chain": ["step 1", "step 2", "..."],
  "really_asking": "the end-of-chain question, one line",
  "anticipated_next": [{"question": "...", "answer_hint": "..."}],
  "tools_recommended": ["web_search","current_time","calculator","url_fetch"],
  "freshness_required": ["..."],
  "confidence_overall": 0.0-1.0,
  "evolution_signals": {
    "user_pattern_observed": "...",
    "good_inference_candidate": true|false
  }
}

VALID EXAMPLE (user asked: "should I email my boss tonight?"):
{
  "hypotheses": [
    {"intent": "seeking permission to wait until morning",
     "probability": 0.6,
     "evidence": ["'should I' framing", "asking late at night"],
     "search_keywords": [],
     "reasoning_type": "pragmatic"},
    {"intent": "wants help drafting the email",
     "probability": 0.25,
     "evidence": ["surface ask"],
     "search_keywords": [],
     "reasoning_type": "deduction"},
    {"intent": "wants reassurance about the boss relationship",
     "probability": 0.15,
     "evidence": ["after-hours urgency"],
     "search_keywords": [],
     "reasoning_type": "abduction"}
  ],
  "implied_chain": ["it is late evening", "boss reads mail tomorrow morning anyway", "so sending now has no upside?", "so waiting is safe?"],
  "really_asking": "is it safe to wait until morning without looking slack?",
  "anticipated_next": [{"question": "what should the email say?", "answer_hint": "short, lead with the ask; offer to draft it"}],
  "tools_recommended": [],
  "freshness_required": [],
  "confidence_overall": 0.6,
  "evolution_signals": {
    "user_pattern_observed": "hedges before workplace decisions",
    "good_inference_candidate": true
  }
}

JSON only. No prose around it. No markdown fences. Just the object.
"""


@dataclass
class InferenceResult:
    hypotheses: list[dict] = field(default_factory=list)
    implied_chain: list[str] = field(default_factory=list)
    really_asking: str = ""
    anticipated_next: list[dict] = field(default_factory=list)
    tools_recommended: list[str] = field(default_factory=list)
    freshness_required: list[str] = field(default_factory=list)
    confidence_overall: float = 0.0
    evolution_signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hypotheses": self.hypotheses,
            "implied_chain": self.implied_chain,
            "really_asking": self.really_asking,
            "anticipated_next": self.anticipated_next,
            "tools_recommended": self.tools_recommended,
            "freshness_required": self.freshness_required,
            "confidence_overall": self.confidence_overall,
            "evolution_signals": self.evolution_signals,
        }


class InferenceEngine:
    # Loop-15 architectural redesign: removed hardcoded
    # IMPLICIT_ASK_TRIGGERS keyword list and periodic_anchor_turns
    # (1, 6, 15, 30, 50, 70). Both were dummy-conversation overfit.
    # SPEC §4.2 says LLM-3 is on-demand; the agentic implementation is
    # that LLM-1 itself decides via the <<sherlock-companions: ...>>
    # tag (already wired). InferenceEngine no longer second-guesses
    # LLM-1; should_fire() now only enforces the cold-start rule
    # (§10.4).

    def __init__(
        self,
        provider: BaseProvider,
        store: MemoryStore,
        system_prompt: str | None = None,
        cold_start_turns: int = 10,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._provider = provider
        self._store = store
        self._prompt = system_prompt or DEFAULT_LLM3_PROMPT
        self._cold_start_turns = cold_start_turns
        self._conf_threshold = confidence_threshold

    def should_fire(
        self,
        *,
        turn_index: int,
        user_text: str,
        topic_changed: bool,
    ) -> bool:
        """Cold-start gate only. The actual decision to fire LLM-3 lives
        upstream in agent.chat() based on LLM-1's <<sherlock-companions>>
        tag. This method exists for backward-compat with the agent flow;
        callers that gate on it preserve the SPEC §10.4 cold-start rule.
        """
        if turn_index < self._cold_start_turns and turn_index != 1:
            return False
        return True

    def infer(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        user_text: str,
        recent_turns: list[ChatMessage],
        llm2_predictions: list[dict] | None = None,
        bypass_cold_start: bool = False,
    ) -> dict:
        # P0-3: when LLM-1 explicitly requested inference via the
        # <<sherlock-companions: infer>> tag, the agent passes
        # ``bypass_cold_start=True`` — honouring the user's "LLM-1 decides"
        # intent. The cold-start gate only applies to non-tag-driven calls.
        if not bypass_cold_start and turn_index < self._cold_start_turns and turn_index != 1:
            return {}

        transcript_lines = [f"{m.role.upper()}: {m.content}" for m in recent_turns]
        transcript = "\n".join(transcript_lines)

        # Provenance ledger — the load-bearing fix for Loop 11.
        # Pull every USER_UTTERANCE memory (these are facts the user has
        # explicitly stated inside the conversation) and pinned SYSTEM-source
        # facts (these are persona-note / domain-hint facts, NOT user-stated).
        # Show them to LLM-3 so it can correctly attribute provenance when
        # the user's current turn is a probe like "did I tell you X" or
        # "you've been calling me Y, did I say that".
        all_mems = self._store.list(conversation_id=conversation_id)
        user_stated = [m.content for m in all_mems if m.type == MemoryType.USER_UTTERANCE]
        system_persona = [
            m.content
            for m in all_mems
            if m.source == MemorySource.SYSTEM and m.type != MemoryType.USER_UTTERANCE
        ]

        # v1.1 (R12): the ledger exists solely to disambiguate USER-STATED
        # facts from SYSTEM-persona notes. With no persona entries, provenance
        # confusion is impossible — skip the whole block (the USER-STATED list
        # alone can run to 40 × 200 chars of pure token cost).
        if system_persona:
            ledger_block = (
                "LEDGER (check before any provenance probe):\n"
                f"USER-STATED ({len(user_stated)}):\n"
                + "\n".join(f"- {u[:200]}" for u in user_stated[-40:])
                + f"\nSYSTEM-PERSONA ({len(system_persona)}) — the user has NOT said these:\n"
                + "\n".join(f"- {s}" for s in system_persona)
                + "\n"
            )
            probe_instruction = (
                "Produce the JSON from your system prompt. On a provenance probe "
                "('did I tell you...'), the top hypothesis MUST distinguish "
                "USER-STATED vs SYSTEM-PERSONA from the ledger."
            )
        else:
            ledger_block = ""
            probe_instruction = "Produce the JSON from your system prompt."

        # v0.4.0: include LLM-2's forward-looking predictions so LLM-3
        # can test / refute / extend them. Only ones above LLM-2's own
        # confidence floor reach here (filtered upstream).
        predictions_block = ""
        if llm2_predictions:
            lines = ["LLM-2 PREDICTIONS — test, refute, or extend:"]
            for p in llm2_predictions[:5]:
                direction = p.get("direction") or p.get("content") or ""
                conf = p.get("confidence")
                evidence = p.get("evidence") or []
                ev_str = "; ".join(evidence[:3]) if evidence else "(no evidence)"
                lines.append(f"- [{conf}] {direction} | evidence: {ev_str}")
            predictions_block = "\n".join(lines) + "\n"

        user_msg = (
            ledger_block
            + ("\n" + predictions_block if predictions_block else "")
            + f"\n--- TRANSCRIPT (most-recent last) ---\n{transcript}\n--- END ---\n\n"
            f"Current user message:\n{user_text}\n\n" + probe_instruction
        )

        messages = [
            ChatMessage(
                role="system",
                content=self._prompt,
                cache_stable_prefix_chars=len(self._prompt),
            ),
            ChatMessage(role="user", content=user_msg),
        ]
        parsed, _resp = chat_json_with_retry(self._provider, messages, want=dict)
        if parsed is None:
            return {}

        result = InferenceResult(
            hypotheses=list(parsed.get("hypotheses") or []),
            implied_chain=[str(x) for x in (parsed.get("implied_chain") or []) if str(x).strip()][
                :6
            ],
            really_asking=str(parsed.get("really_asking") or "").strip()[:300],
            anticipated_next=[
                x for x in (parsed.get("anticipated_next") or []) if isinstance(x, dict)
            ][:2],
            tools_recommended=list(parsed.get("tools_recommended") or []),
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

    def review_search(
        self,
        *,
        topic: str,
        hypotheses: list[dict],
        results: list[dict],
        round_index: int,
        max_rounds: int,
    ) -> dict:
        """v0.7: LLM-3 self-evaluates ONE round of inference-search results and
        decides whether to keep digging. Returns a dict with keys: recent,
        fleshes_out, right_query, worth_saving, need_more, next_queries (≤2),
        note. Best-effort — on any failure returns a conservative "stop".
        """
        stop = {"need_more": False, "worth_saving": True, "next_queries": [], "note": ""}
        try:
            hyp_txt = (
                "; ".join(str(h.get("intent", "")) for h in (hypotheses or [])[:3]) or "(none)"
            )
            res_txt = (
                "\n".join(
                    f"- {r.get('title','')} — {r.get('url','')}: "
                    f"{(r.get('content') or r.get('snippet') or '')[:200]}"
                    for r in (results or [])[:8]
                )
                or "(no results)"
            )
            prompt = (
                "You are LLM-3 reviewing ONE round of web-search results you "
                "requested to back your inference. Judge honestly.\n\n"
                f"Your active hypotheses: {hyp_txt}\n"
                f"Search topic this round: {topic}\n"
                f"Round {round_index} of at most {max_rounds}.\n\n"
                f"Results:\n{res_txt}\n\n"
                "Return STRICT JSON only:\n"
                "{\n"
                '  "recent": true|false,\n'
                '  "fleshes_out": true|false,\n'
                '  "right_query": true|false,\n'
                '  "worth_saving": true|false,\n'
                '  "need_more": true|false,\n'
                '  "next_queries": ["...", "..."],\n'
                '  "note": "one short line"\n'
                "}\n"
                "Set need_more=true ONLY if a specific, nameable gap remains "
                "(give ≤2 refined next_queries). Otherwise need_more=false and "
                "next_queries=[]. JSON only — no prose, no fences."
            )
            parsed, _resp = chat_json_with_retry(
                self._provider,
                [
                    ChatMessage(role="system", content=RESEARCH_DIRECTOR_PERSONA),
                    ChatMessage(role="user", content=prompt),
                ],
                want=dict,
            )
            if not isinstance(parsed, dict):
                return stop
            parsed.setdefault("need_more", False)
            parsed.setdefault("worth_saving", True)
            nq = parsed.get("next_queries") or []
            parsed["next_queries"] = [str(q).strip() for q in nq if str(q).strip()][:2]
            return parsed
        except Exception:
            return stop

    def generate_meta_questions(
        self,
        *,
        topic: str,
        queries: list[str],
        findings_digest: str,
        round_index: int,
        max_questions: int = 5,
        lang_hint: str = "",
        usage_sink=None,
        today: str = "",
    ) -> list[str]:
        """v0.7 Phase 3: from deep-research round 3, LLM-3 GENERATES the
        meta-cognition questions that drive the next round — pushing depth and
        breadth the user wouldn't think of (adjacent topics, second-order
        effects, contradicting sources, overlooked stakeholders/timeframes).

        ``lang_hint`` (the user's original request) makes the QUESTIONS come out
        in the user's language, since they are surfaced in the UI/docs.

        Returns a list of question strings. Best-effort — falls back to ``[]``
        so the caller can use its fixed question set.
        """
        try:
            qs_txt = "; ".join(str(q) for q in (queries or [])[:5]) or "(none)"
            lang_line = (
                f"Write the questions in the SAME language as this user request: "
                f"«{lang_hint[:200]}».\n"
                if lang_hint
                else ""
            )
            prompt = (
                "You are LLM-3 steering a DEEP research loop. We are at round "
                f"{round_index}. Core topic: {topic}\n"
                + (today + "\n" if today else "")
                + f"Recent queries: {qs_txt}\n"
                f"What we've found so far (digest):\n{findings_digest[:1500]}\n\n"
                "Generate the META-COGNITION QUESTIONS that should drive the "
                "NEXT round — questions that DEEPEN and BROADEN beyond the "
                "obvious: adjacent/peripheral topics, second-order effects, "
                "contradicting evidence, who/what/when is being overlooked, and "
                "whether the current direction is even the right one. Ask things "
                "the user would not think to ask.\n"
                "At least ONE question must actively seek DISCONFIRMING evidence "
                "for the strongest current finding.\n"
                + lang_line
                + f"Return STRICT JSON: a list of {max_questions} short question "
                'strings, e.g. ["...", "..."]. JSON only — no prose, no fences.'
            )

            def _account(r):
                if usage_sink:
                    usage_sink(_usage_or_estimate(r, prompt))

            parsed, _resp = chat_json_with_retry(
                self._provider,
                [
                    ChatMessage(role="system", content=RESEARCH_DIRECTOR_PERSONA),
                    ChatMessage(role="user", content=prompt),
                ],
                want=None,
                on_usage=_account,
            )
            out: list[str] = []
            if isinstance(parsed, list):
                out = [str(q).strip() for q in parsed if str(q).strip()]
            elif isinstance(parsed, dict):
                seq = parsed.get("questions") or parsed.get("meta_questions") or []
                out = [str(q).strip() for q in seq if str(q).strip()]
            return out[:max_questions]
        except Exception:
            return []

    def plan_search(
        self,
        *,
        topic: str,
        purpose_hint: str = "",
        user_lang: str = "",
        default_languages: list[str] | None = None,
        max_queries: int = 6,
        usage_sink=None,
        today: str = "",
    ) -> list[dict]:
        """v0.8: produce a MULTILINGUAL, keyword-style search plan for deep
        research. Returns ``[{"lang": code, "keywords": clean_query}, ...]`` —
        ≥2 languages, **never just the user's language** (output language is
        decoupled from search languages). Best-effort: on failure / LLM-3
        disabled, returns a deterministic multilingual fallback.
        """
        from sherlock.tools.web_search import clean_query

        # Honest single-entry fallback: without an LLM we cannot translate the
        # topic, and duplicating it under a fake "en" label adds no diversity —
        # it just runs the identical query twice.
        fallback = [{"lang": (user_lang or "und"), "keywords": clean_query(topic) or topic}]
        try:
            langs_line = (
                f"Preferred languages (guidance): {', '.join(default_languages)}.\n"
                if default_languages
                else ""
            )
            prompt = (
                "You are LLM-3 planning a MULTILINGUAL web-search sweep for deep research.\n"
                + (today + "\n" if today else "")
                + f"Topic: {topic}\n"
                f"Why the user asked (purpose): {purpose_hint or topic}\n"
                f"User's language: {user_lang or 'unknown'} — this is the OUTPUT language "
                "ONLY; do NOT restrict the SEARCH to it.\n"
                + langs_line
                + "Choose the languages whose web has the MOST relevant data for THIS topic "
                "(e.g. Japan travel → Japanese + Korean + English). Use AT LEAST 2 languages; "
                "never just the user's language. For each language give SHORT keyword queries "
                "(2-5 words, NO particles/punctuation/sentences).\n"
                "At least ONE query must target counter-evidence — a skeptical angle "
                "(problems / criticism / limitations of the topic).\n"
                f"Return STRICT JSON: a list of up to {max_queries} objects, e.g. "
                '[{"lang":"ja","keywords":"..."},{"lang":"ko","keywords":"..."}]. '
                "JSON only — no prose, no fences."
            )

            def _account(r):
                if usage_sink:
                    usage_sink(_usage_or_estimate(r, prompt))

            parsed, _resp = chat_json_with_retry(
                self._provider,
                [
                    ChatMessage(role="system", content=RESEARCH_DIRECTOR_PERSONA),
                    ChatMessage(role="user", content=prompt),
                ],
                want=None,
                on_usage=_account,
            )
            seq = (
                parsed
                if isinstance(parsed, list)
                else (parsed.get("queries") if isinstance(parsed, dict) else None)
            )
            out: list[dict] = []
            for item in seq or []:
                if isinstance(item, dict):
                    kw = clean_query(str(item.get("keywords") or item.get("query") or "").strip())
                    lang = str(item.get("lang") or item.get("language") or "").strip() or "und"
                    if kw:
                        out.append({"lang": lang, "keywords": kw})
                elif isinstance(item, str) and item.strip():
                    out.append({"lang": "und", "keywords": clean_query(item)})
            if not out:
                return fallback
            # If the model ignored the multi-language instruction, add the raw
            # topic as one extra query (honestly unlabelled) when it's distinct.
            if len({o["lang"] for o in out}) < 2:
                kw = clean_query(topic) or topic
                if kw and kw not in {o["keywords"] for o in out}:
                    out.append({"lang": "und", "keywords": kw})
            # Slice preserving language breadth: round-robin across languages so
            # a long single-language head can't crowd the others out.
            by_lang: dict[str, list[dict]] = {}
            for o in out:
                by_lang.setdefault(o["lang"], []).append(o)
            interleaved: list[dict] = []
            buckets = list(by_lang.values())
            i = 0
            while len(interleaved) < len(out):
                added = False
                for b in buckets:
                    if i < len(b):
                        interleaved.append(b[i])
                        added = True
                if not added:
                    break
                i += 1
            return interleaved[:max_queries]
        except Exception:
            return fallback


def _usage_or_estimate(resp, prompt: str):
    """Provider-reported usage, or a count_tokens estimate when the provider
    (callable/wrapper — the flagship BYO path) reports none. Keeps deep-research
    token accounting non-zero for every stage."""
    u = getattr(resp, "usage", None)
    if u is not None and (
        getattr(u, "prompt_tokens", 0) or getattr(u, "completion_tokens", 0) or 0
    ):
        return u
    try:
        from sherlock.budget import count_tokens
        from sherlock.providers.base import TokenUsage

        pin = count_tokens(prompt or "")
        pout = count_tokens(getattr(resp, "text", "") or "")
        return TokenUsage(prompt_tokens=pin, completion_tokens=pout, total_tokens=pin + pout)
    except Exception:
        return u


# v1.0: the lenient parse ladder lives in sherlock.jsonish (shared with the
# summarizer). Re-exported under the old names because agent.py and tests
# import them from this module.
_loads_lenient = loads_lenient
_extract_balanced = extract_balanced
_safe_parse_json = safe_parse_json
