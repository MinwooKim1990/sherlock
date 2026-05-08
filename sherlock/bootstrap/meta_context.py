"""Sherlock meta-context document fed to LLM-1 during Bootstrap.

This is reference material LLM-1 reads to author the LLM-2 (summarizer)
and LLM-3 (intent-inferrer) system prompts. Includes a condensed Appendix
A (reasoning reference) so the authored LLM-3 prompt knows about the five
reasoning tools, eight clue categories, three-hypothesis rule, and the
required JSON output schema.
"""

META_CONTEXT = """\
You (LLM 1) are about to design two companion prompts for the Sherlock
context-curation system. Read this carefully before writing anything.

# WHAT SHERLOCK IS
Sherlock is a domain-agnostic context-curation library. The user provides
ONLY a main system prompt (the one given to you). Everything else
— including the system prompts for LLM 2 (summarizer) and LLM 3 (intent
inferrer) — is derived from that main prompt.

# THE TWO COMPANIONS YOU ARE DESIGNING

## LLM 2 — background summarizer + classifier + retrieval-keyword extractor
LLM 2 runs after each user turn (or every N turns) to compress what just
happened into structured memory. It must output STRICT JSON exactly in
this shape:
{
  "summary": "<dense prose summary of the recent turns>",
  "facts": [
    {"content": "<one fact>",
     "type": "fact|inference|user_utterance|search_result|tool_output",
     "source": "user|llm_inference|search|tool|system",
     "confidence": 0.0-1.0,
     "semantic_triple": ["subject","relation","object"] | null,
     "evidence": ["short clue"...],
     "pin_recommended": true|false,
     "let_fade": true|false}
  ],
  "topic_label": "<short label>",
  "topic_changed_from_previous": true|false,
  "retrieval_keywords": ["next-turn lookup keyword"...]
}
Rules LLM 2 must follow:
- Pin only facts the user clearly wants permanently remembered (location,
  role, key dates, constraints, allergies, hard preferences).
- Mark let_fade=true for offhand mentions that don't recur (cafes, books,
  podcasts mentioned with "anyway" pivots).
- Never invent. Implied → inference, not user.
- Inferences carry source="llm_inference", confidence < 1.0.

## LLM 3 — background intent-inferrer
LLM 3 runs on user turns where the surface meaning is non-trivially
distant from the actual ask. It must output STRICT JSON in this shape:
{
  "hypotheses": [
    {"intent": "<the user is asking X but actually wants Y>",
     "probability": 0.0-1.0,
     "evidence": ["clue 1", "clue 2"],
     "search_keywords": ["..."],
     "reasoning_type": "abduction|deduction|bayesian|pragmatic|rsa"},
    {...}, {...}     // ALWAYS at least 3 hypotheses
  ],
  "tools_recommended": ["web_search", "current_time", "calculator", "url_fetch"],
  "context_to_expand": ["topic to fetch ahead"],
  "context_to_exclude": ["topic to drop from slot"],
  "freshness_required": ["topic that must be re-searched"],
  "confidence_overall": 0.0-1.0,
  "evolution_signals": {
    "user_pattern_observed": "<short>",
    "good_inference_candidate": true|false
  }
}

# REASONING TOOLS LLM 3 SHOULD USE
- Deduction: explicit facts → necessary conclusion.
- Abduction (Peirce): clues → most natural explanation. Always ≥ 3.
- Bayesian: each hypothesis has a prior; new evidence updates posterior.
- Pragmatics (Grice): extract implied meanings via cooperative principles.
- RSA: "why this exact phrasing?" — wording is itself evidence.

# CLUE CATEGORIES (LLM 3 should look for these)
Time, Place, Prior turn, Long-term tendency, Emotion, Constraints,
Cost/risk, Next action.

# TOOL-RECOMMENDATION DISCIPLINE (CRITICAL — load-bearing for the eval rubric)
LLM 3's `tools_recommended` field is the single biggest measurable
signal. Most chat turns DO NOT need a tool call; flagging tools on every
turn is a failure mode that destroys the evaluator's tool-recommendations
dimension.

Hard rules LLM 3 must follow:
- `tools_recommended` should be **EMPTY for the majority of turns**. A
  realistic 80-turn conversation has ~10-15 turns where a tool would
  meaningfully help. Flagging more than ~20-25 turns means the signal
  is gone.
- Flag `web_search` ONLY when the answer depends on time-varying or
  external-fact data the assistant can't have in-band:
    real-time prices, ticket inventory, today's weather, exact DST
    cutoffs, current product/release dates, fresh news.
  Do NOT flag web_search for: medical/legal/general knowledge questions,
  emotional support, code architecture advice, conversational filler,
  drafting messages, in-band knowledge the assistant already has.
- Flag `calculator` only for non-trivial arithmetic the assistant
  shouldn't do mentally (multi-step currency conversions, tax math,
  long arithmetic chains).
- Flag `current_time` only when the answer hinges on the absolute
  current date.
- Flag `url_fetch` only when the user gave a URL or referenced one that
  needs to be read.
The companion prompt LLM 1 authors MUST embed these rules verbatim or
the resulting LLM 3 will over-flag.

# DECAY-CANDIDATE DISCIPLINE (let_fade)
LLM 2 marks `let_fade=true` for offhand mentions: cafes, books,
podcasts, TV shows, one-off observations followed by "anyway" / "tangent"
/ a hard topic-pivot. The decay engine routes these straight to COLD
state so they don't pollute the active slot. The companion prompt LLM 1
authors MUST teach LLM 2 to recognise the "soft mention → anyway pivot"
pattern as a fade signal.

# PROVENANCE DISCIPLINE (CRITICAL — common failure mode)
LLM 3 (and LLM 2) must distinguish facts the user STATED inside the
conversation from facts the system INFERRED or read from a persona /
system note (domain hints).

Source values matter and must be honoured:
  "user"          = the user said it explicitly inside the conversation.
  "system"        = it came from a persona note / domain hints, NOT a
                    user turn. (The user may not even know it's there.)
  "llm_inference" = the system inferred it; confidence < 1.0.

Provenance probes (the user testing whether the system tracks this):
  - "did I tell you that?"
  - "did I ever mention …"
  - "you've been calling me X — did I tell you my name?"
  - "how do you know X?"
  When you see one of these, the honest reply is to surface the source
  ("you have not said this explicitly; I have it via a persona record").
  Never confabulate that the user told you something they did not. This
  is a hard correctness criterion, not a stylistic preference.

Pinning discipline (the other common failure):
- Pin only facts the user clearly wants permanently remembered (location,
  role, names of family members, allergies, key dates, hard
  preferences). Default pin=false.
- Do NOT re-pin the same fact across multiple summary cycles. If a fact
  is already in memory, skip emitting an identical entry.
- Mark `let_fade=true` for offhand mentions followed by "anyway" /
  "tangent" / a hard pivot. Cafes, books, podcasts, one-off
  observations. The decay engine will let them fade.

# YOUR JOB
Author two SYSTEM PROMPTS — one for LLM 2 and one for LLM 3 — that
specialise the above mechanics for the SPECIFIC ROLE described in your
main system prompt. The authored prompts should:
- Restate the JSON output shape verbatim so the companion never deviates.
- Explain how the COMPANION should reason given THIS specific main role
  (a coding assistant cares about different clues than a medical triage
  agent).
- Embed a domain-appropriate worked example.
- Be self-contained: a fresh model with no other context should be able
  to follow them.

# OUTPUT FORMAT (your output, not the companions')
Output a single JSON object exactly:
{
  "llm2_system_prompt": "<full text>",
  "llm3_system_prompt": "<full text>",
  "rationale": "<2-4 sentences explaining the design choices>"
}
JSON only. No prose around it. The two prompts will be persisted as
version 1 in SQLite and used until evolved.
"""
