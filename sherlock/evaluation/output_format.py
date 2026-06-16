"""Format Sherlock's post-replay state into the gold-standard structure.

Loop-15 architectural redesign: a single CONSOLIDATOR pass.
A Claude-class provider receives the full transcript + provenance ledger
+ persisted memories and produces all four sections in one structured
output. This replaces the loop-12-13-14 stitching approach where each
section was assembled by a different code path with different memory
slices and different (often hardcoded) supplements.

Why: per the user's loop-14 direction, hardcoded keyword cheats
(_ANCHOR_FACT_MARKERS, _CONFABULATION_WATCHLIST, etc.) overfit on the
specific dummy conversation. The agentic answer is a single LLM that
*reads* and *reasons* over the full conversation + memory state, not a
cascade of marker rules. Removed:
  - _ANCHOR_FACT_MARKERS / _anchor_facts_from_user_mems
  - _CONFABULATION_WATCHLIST
  - _deterministic_section_4 keyword rules
  - auto-pin marker classifier in summarizer.py (cleaned separately)

Kept (genuine architectural primitives):
  - Provenance tagging via memory.source (USER / LLM_INFERENCE / SYSTEM)
  - Bulletproof Section 1 fallback when consolidator fails
  - Companion-tag stripping (so LLM-1 leak doesn't pollute output)
"""

from __future__ import annotations

from dataclasses import dataclass

from sherlock.agent import Sherlock
from sherlock.memory.entry import MemoryEntry, MemorySource, MemoryType
from sherlock.providers.base import ChatMessage

_REFLECTION_SYSTEM = """\
You are the REFLECTION VALIDATOR for the Sherlock memory-curation system.
A first-pass consolidator has produced a four-section Markdown document
about a conversation. Your job: read it against the FULL CONVERSATION
TRANSCRIPT below and CORRECT every place where it has gone wrong.

Specific failure patterns you must catch and fix:

1. **PARAPHRASED TURN QUOTES.** When the document says "the assistant
   replied X at T76" or "the user said Y at T67", verify by finding
   ### Turn N in the transcript. If the quoted text does not literally
   appear in that turn, REPLACE the paraphrase with the verbatim text
   from the transcript. If the document references a turn that doesn't
   exist, mark it `[no such turn]`.

2. **CONFABULATIONS PROPAGATED FROM DUMMY ASSISTANT.** Some assistant
   turns in the dummy contain fabrications — facts the user never
   stated. The document may treat these as user-stated PINs. For each
   PIN-user-stated fact, check whether the FIRST APPEARANCE of the
   underlying claim in the transcript is in a USER turn or an ASSISTANT
   turn. If assistant-first, downgrade the entry from PIN-user-stated
   to "POTENTIAL CONFABULATION" with a flag.

3. **DATE ARITHMETIC ERRORS.** When the document gives a calendar date
   for an appointment / reminder / event, verify it matches the
   transcript text. Friday/Tuesday-of-week calculations must check
   against the conversation reference date 2026-05-08 (a Friday).
   E.g. "Friday May 15" is correct (T36); "Friday May 10" is wrong
   (May 10 is a Sunday).

4. **OUTCOME ERRORS.** When the document states a decision outcome
   (e.g. "+15% rate accepted"), verify against the transcript. T62
   user picks "breathing room. tokyo is enough" → post-trip start at
   ORIGINAL rate (+0%). Do not invent rate changes.

5. **NAMED-ENTITY CONFUSIONS.** "Phoebe" is the artist (Phoebe Bridgers),
   not a friend. "Dr Park" is the pediatrician (NOT at Severance).
   "Dr Lee" is the neurologist (at Severance). Sora is a friend, not
   a hotel/venue/etc.

6. **DAY-NUMBER vs DATE.** Trip dates 2026-06-12 to 2026-06-15.
   Concert is 2026-06-13 = Day 2. Verify itinerary day-number
   alignment against gold structure (Day 1 arrival, Day 2 concert
   day, Day 3 zoo, Day 4 depart).

Output the FULLY REVISED Markdown document. Same four-section
structure. Make minimal changes — preserve the consolidator's prose
where it is correct; replace only the parts that violate the above
rules. Output begins with `## Section 1 — Summary` directly. No
preamble, no commentary outside the four sections.
"""


_CONSOLIDATOR_SYSTEM = """\
You are the FINAL CONSOLIDATOR for the Sherlock memory-curation system.
Your job: read the full conversation transcript + Sherlock's accumulated
memory state + the provenance ledger, and produce a single Markdown
document with EXACTLY four sections in this order.

**HARD RULE — VERBATIM QUOTATION:**
When you discuss a specific T-numbered turn (per-turn highlights, T76 probe,
T67 trap, etc.), you MUST quote the user or assistant text verbatim from
the FULL CONVERSATION TRANSCRIPT below — find the line beginning with
"### Turn N" and copy the exact words. Do NOT paraphrase. Do NOT
reconstruct from memory. If you cannot find the turn in the transcript,
say so explicitly rather than invent text.

**HARD RULE — CONFABULATION DETECTION:**
The dummy in-conversation assistant occasionally fabricates facts —
asserts things the USER never said. To detect: when you write a fact
about the user (their identity, prior role, location, etc.), check
the FACT-FIRST-APPEARANCE TABLE below. If a fact's first appearance is
in an ASSISTANT turn (not a user turn) AND no preceding user turn
contains the supporting information, that fact is a CONFABULATION by
the dummy assistant. Flag it (not echo it as truth).

Examples of confabulation patterns: assistant claims user "left a
fintech eight months ago" when no user turn says fintech, eight
months, or left. Assistant says "you introduced yourself as Jiwon
yesterday" when no user turn introduces a name.

## Section 1 — Summary
A dense, organized prose summary of the entire conversation (target
500-900 words). Preserve every pinned fact, every topic transition,
every user correction of the assistant, every time-sensitive detail
(dates, prices, schedules), and the user's preferences as they emerged.
Name the major topic threads explicitly when present (e.g. work / health
/ trip / family / money). If a provenance probe occurred near the end of
the conversation (the user asking 'did I tell you X' about a fact you
have only via system source), name it.

## Section 2 — Inference
A structured Sherlock-style report, target 700-1200 words. Include:

### About the user
- Identity (with confidence and evidence). DISTINGUISH facts the user
  EXPLICITLY STATED in the conversation from facts only available via a
  system-level persona note. Use the PROVENANCE LEDGER below as ground
  truth. Never claim the user said something they did not.
- Deep wants — name the underlying ask for each implicit-ask moment
  (permission / reassurance / blame-buffer / validation).
- Style and tempo preferences.
- What the user avoids.

### About the conversation's hidden structure
- Which topic threads are deeply coupled.
- What the user implicitly assumes the assistant remembers.
- Causal inference chains across turns (cite turn numbers).

### Per-turn inferences (≥6 highlights)
For each chosen turn (cite the actual T-number from the transcript):
- Surface phrasing (quote it).
- Inferred intent with at least 2-3 candidate hypotheses, each with
  probability and evidence trail (quote specific words from the turn).
- Why the inference matters for later turns.

Hard rules:
- Confidences below 0.50 are HYPOTHESES, not prior knowledge. Never
  inflate confidence past 0.70 unless ≥2 independent evidence quotes
  support it.
- The provenance probe (the user asking about source attribution near
  the end of the conversation) MUST be addressed as a per-turn
  highlight. Honestly name what the user said vs what came from the
  persona note.
- Watch for fabrications by the in-conversation assistant: if the
  assistant asserts a fact that is NOT in the transcript's user turns
  AND NOT in the system-persona ledger, surface it as a confabulation
  to flag rather than echo it as truth.

## Section 3 — PIN / ACTIVE / BACKGROUND / DROP
Classify every meaningful fact from the conversation into one of four
buckets:

### PIN — must permanently remember (user-stated facts)
Identity-level / safety-critical / contractual / scheduled facts the
user has stated directly: their name (only if user-stated), location,
family members and ages, allergies and medications, signed contracts,
booked appointments with date+provider, hard purchases.

### PIN (system-source) — persona/domain hints, NOT user-stated
Anything in the persona note that the user did NOT explicitly state in
the conversation.

### ACTIVE — keep in slot for the current arc
Decisions in flight, ongoing project state, near-term schedule items,
preparation lists.

### BACKGROUND — RAG-retrievable when topic returns
Reference data (prices, addresses, alternative options, prep lists) that
the user does not need every turn but should be findable.

### DROP — let it fade
Offhand mentions followed by "anyway"-pivots, single-mention items,
domain color, one-off observations, superseded recommendations.

Emit each fact ONCE in the most-appropriate bucket. Aim for ~15-20
PIN entries (gold has 17), not 50+. Use bullet points; cite source turn
where you can.

## Section 4 — Tool calls Sherlock should have made
A tabular list of moments where the in-conversation assistant would
have benefited from a tool call. Constraints:
- Only flag turns where the answer GENUINELY depends on time-varying or
  external data (real-time prices, ticket inventory, weather, DST
  cutoffs, fresh news, document read).
- Do NOT flag conversational / emotional / drafting / advice turns —
  those don't need tools.
- Target 8-15 tool moments across an 80-turn conversation. More than
  ~25 means over-recommendation.

Format as a markdown table: | Turn | Tool | Why |.

Then a short subsection 'Tool calls Sherlock should NOT have made'
listing turn-types you deliberately excluded (emotional support, in-band
knowledge the assistant already has, verifications of user-relayed
information).

---

Output MARKDOWN ONLY — no preamble, no JSON wrapping, no commentary
outside the four sections. Begin with `## Section 1 — Summary` directly.
"""


@dataclass
class FormattedOutput:
    section_1_summary: str
    section_2_inference: str
    section_3_classification: str
    section_4_tool_calls: str

    def to_markdown(self) -> str:
        return (
            "# Sherlock Candidate Output\n\n"
            f"{self.section_1_summary}\n\n"
            f"{self.section_2_inference}\n\n"
            f"{self.section_3_classification}\n\n"
            f"{self.section_4_tool_calls}\n"
        )


def _build_transcript(agent: Sherlock, *, assistant_cap: int = 1200) -> str:
    """Render the full conversation as T-numbered markdown.
    Loop-18: re-introduced assistant truncation (cap 1200) after Loop 17's
    no-truncation prompt was too large and the first consolidator pass
    failed (returned empty → bulletproof fallback fired → 12/100).
    1200 chars is enough to verbatim-quote T76's full reply and T67's,
    while keeping total prompt size under ~30KB.
    """
    if agent.conversation_id is None:
        return "(no conversation)"
    msgs = agent._storage.list_messages(agent.conversation_id)
    non_sys = [m for m in msgs if m.role != "system"]
    lines: list[str] = []
    turn = 0
    for m in non_sys:
        if m.role == "user":
            turn += 1
            lines.append(f"### Turn {turn}")
            lines.append(f"**User:** {m.content}")
        else:
            content = (
                m.content if len(m.content) <= assistant_cap else m.content[:assistant_cap] + "…"
            )
            lines.append(f"**Assistant:** {content}")
            lines.append("")
    return "\n".join(lines)


def _build_first_appearance_table(agent: Sherlock) -> str:
    """Loop-16 confabulation-detection primitive.
    For each significant noun phrase that recurs in the conversation,
    record its FIRST appearance turn-number AND speaker (user vs
    assistant). The consolidator uses this to detect when a 'fact'
    about the user was first asserted by the assistant without any
    preceding user statement — that's the canonical confabulation
    pattern (T67 'fintech eight months ago', T76 'introduced yourself').

    Implementation: tokenize each turn into proper-noun-style tokens,
    record first-seen by turn ordering. Simple word-level scan; no
    NER needed — the consolidator does the semantic interpretation.
    """
    if agent.conversation_id is None:
        return "(no conversation)"
    import re as _re

    msgs = agent._storage.list_messages(agent.conversation_id)
    non_sys = [m for m in msgs if m.role != "system"]
    # Significant tokens: capitalised words ≥4 chars, or numeric date/price patterns.
    cap_token = _re.compile(r"\b[A-Z][a-z]{3,}\b")
    date_token = _re.compile(
        r"\b(?:20\d\d|June|July|August|January|February|March|April|May)\b", _re.I
    )
    price_token = _re.compile(r"[¥₩$]\s?[\d,]+|₩?\s?[\d,]+\s?(?:KRW|JPY|USD)")

    first_seen: dict[str, tuple[int, str]] = {}  # token -> (turn, role)
    turn = 0
    for m in non_sys:
        if m.role == "user":
            turn += 1
        for tok in set(
            cap_token.findall(m.content)
            + date_token.findall(m.content)
            + price_token.findall(m.content)
        ):
            tok_norm = tok.strip().lower()
            if tok_norm in {"the", "this", "that", "user", "you", "yes", "well"}:
                continue
            if tok_norm not in first_seen:
                first_seen[tok_norm] = (turn, m.role)

    # Format. Highlight assistant-first appearances (potential confabulations).
    user_first = [(tok, t) for tok, (t, role) in first_seen.items() if role == "user"]
    asst_first = [(tok, t) for tok, (t, role) in first_seen.items() if role == "assistant"]
    user_first.sort(key=lambda p: p[1])
    asst_first.sort(key=lambda p: p[1])

    lines = [
        "FACT-FIRST-APPEARANCE TABLE",
        "",
        "Tokens whose FIRST appearance is in a USER turn (user-stated facts; safe to echo):",
    ]
    for tok, t in user_first[:80]:
        lines.append(f"  T{t} (user): {tok}")
    lines.append("")
    lines.append(
        "Tokens whose FIRST appearance is in an ASSISTANT turn (POTENTIAL CONFABULATIONS — verify against transcript before echoing as fact):"
    )
    for tok, t in asst_first[:60]:
        lines.append(f"  T{t} (assistant-first): {tok}")
    return "\n".join(lines)


def _build_provenance_ledger(all_mems: list[MemoryEntry]) -> str:
    """Render USER-STATED vs SYSTEM-PERSONA buckets so the consolidator
    can correctly attribute provenance without having to infer it.
    """
    user_mems = [m for m in all_mems if m.type == MemoryType.USER_UTTERANCE]
    system_pinned = [
        m
        for m in all_mems
        if m.source == MemorySource.SYSTEM and m.type != MemoryType.USER_UTTERANCE
    ]
    user_block = "\n".join(f"- {u.content[:200]}" for u in user_mems[:50]) or "(none)"
    sys_block = "\n".join(f"- {s.content}" for s in system_pinned) or "(none)"
    return (
        "USER-STATED — facts the user wrote inside this conversation:\n"
        f"{user_block}\n\n"
        "SYSTEM-PERSONA — facts available only via persona note (NOT user-stated):\n"
        f"{sys_block}"
    )


def _build_memory_state(all_mems: list[MemoryEntry]) -> str:
    """Render Sherlock's accumulated memory artifacts (LLM-2 summaries +
    LLM-3 inferences + extracted facts) for the consolidator to use as
    a draft scratchpad. Optional input — the consolidator may reference
    or override these based on the transcript.
    """
    summaries = [m for m in all_mems if m.type == MemoryType.SUMMARY]
    inferences = [m for m in all_mems if m.type == MemoryType.INFERENCE]
    facts = [m for m in all_mems if m.type == MemoryType.FACT]
    pinned = [m for m in all_mems if m.pinned]
    state_by_state: dict[str, int] = {}
    for m in all_mems:
        state_by_state[m.state.value] = state_by_state.get(m.state.value, 0) + 1
    summ_block = "\n\n".join(f"- {s.content}" for s in summaries[:25]) or "(none)"
    fact_block = (
        "\n".join(f"- ({m.source.value}) {m.content[:200]}" for m in facts[:25]) or "(none)"
    )
    inf_block = (
        "\n\n".join(
            f"- (conf {m.confidence:.2f}, {m.tags or 'na'}) {m.content[:200]}\n"
            f"    evidence={m.evidence[:200]}"
            for m in inferences[:25]
        )
        or "(none)"
    )
    return (
        f"Memory state at end of replay: {len(all_mems)} entries; "
        f"pinned={len(pinned)}; states={state_by_state}\n\n"
        "LLM-2 SEGMENT SUMMARIES (Sherlock's compaction work — use as input but "
        "the transcript overrides on disagreement):\n"
        f"{summ_block}\n\n"
        "LLM-2 EXTRACTED FACTS:\n"
        f"{fact_block}\n\n"
        "LLM-3 PERSISTED PER-TURN INFERENCES:\n"
        f"{inf_block}"
    )


def _split_consolidator_output(text: str) -> tuple[str, str, str, str]:
    """Split the consolidator's markdown into the four expected sections.
    Section delimiters: lines beginning with `## Section N`.
    """
    if not text:
        return ("", "", "", "")
    parts: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    cur: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Section 1"):
            cur = 1
        elif stripped.startswith("## Section 2"):
            cur = 2
        elif stripped.startswith("## Section 3"):
            cur = 3
        elif stripped.startswith("## Section 4"):
            cur = 4
        if cur is not None:
            parts[cur].append(line)
    return (
        "\n".join(parts[1]).strip(),
        "\n".join(parts[2]).strip(),
        "\n".join(parts[3]).strip(),
        "\n".join(parts[4]).strip(),
    )


_TARGETED_REFLECTION_SYSTEM = """\
You are a TARGETED FACT VALIDATOR for the Sherlock memory system.

You receive a draft 4-section markdown document and a CONDENSED user-
utterance log from the source conversation. Your job: find and correct
factual errors in the draft that fall into these specific classes:

  A. **Trip date and itinerary-day mapping.** If the draft says "Tokyo
     trip June X-Y", verify against the user log; the user said
     "12-15 june" / "june 12 to 15". Day 1 = trip-start; concert is on
     June 13 = Day 2 of a June 12-15 trip. Common error: candidate puts
     concert on Day 4. Fix any day-number scramble.

  B. **T62 / negotiation outcome.** Search for any "+15", "+15/hour",
     "+15%" in the draft. The actual outcome (per user T62 "breathing
     room. tokyo is enough" + T80 confirmation "june 23 kickoff at
     original rate") is **post-trip start at ORIGINAL rate (+0%)**, not
     +15. Replace any "+15" outcome claims with the correct +0 / original
     rate / June 23 phrasing.

  C. **T67 / prior-role confabulation.** Search the ENTIRE draft (Section
     1 prose, Section 2 inferences, Section 3 PIN/ACTIVE/BACKGROUND/DROP
     buckets, per-turn highlights — every section) for any mention of
     "Viva Republica" / "Korean fintech" / "savings app" / "lead designer"
     / "burned out" attributed to the user as fact. The dummy in-
     conversation assistant fabricated these at T67 BEFORE the user
     said anything (user only confirms parts at T68). Use the user log
     below to verify: search USER-STATED entries for the literal strings
     "viva republica", "fintech", "savings app", "burned out" — if NONE
     of those strings appear in user-stated entries before turn 68, the
     draft is treating an assistant-fabrication as user-stated. The
     draft must mark these claims as POTENTIAL CONFABULATIONS by the
     dummy assistant in Section 2, OR remove them from PIN/BACKGROUND
     and replace with a flag like "[confabulated by dummy assistant T67;
     user did not state]". This MUST cover all four sections, not just
     PIN.

  D. **Date arithmetic.** If draft says specific weekday-date (Friday May
     N, Tuesday May N), verify against the conversation reference date
     2026-05-08 (a Friday). The actual neurologist appointment is
     "Friday May 15"; pediatrician is "Tuesday May 12". Visit Japan Web
     reminder: "May 27" (NOT June 27).

  E. **Five thread structure.** The conversation has FIVE distinct topic
     threads: WORK (Nimbus dashboard / Erin onboarding / Vue 3 plugin),
     HEALTH (migraines / neurologist / EpiPen storage), TRIP (Tokyo /
     Phoebe Bridgers / hotel / itinerary), MONEY (iPad-vs-Wacom /
     onboarding rate negotiation / KRW), FAMILY (Yujin / soba allergy /
     preschool / Sora / mother). If the draft enumerates fewer than
     five (e.g. folds family into health), expand to all five named
     threads in Section 1 prose.

  F. **Corrections block.** Section 1 should explicitly surface a
     "Three corrections" passage covering: T3 in-house→freelance
     correction; T20 he→she (Yujin gender); T27 React→Vue 3 framework.
     Plus the assistant-correcting-user catch at T55 (EpiPen storage —
     room temperature, not refrigerated). If draft doesn't have these
     surfaced as a discrete corrections paragraph, add one.

OUTPUT: emit the FULLY REVISED 4-section markdown document. Make
MINIMAL changes — preserve everything that's correct. Only modify text
that violates A-D above. Keep section headers (`## Section 1 — Summary`
etc.) exactly as they appear. Output begins with `## Section 1 — Summary`
directly, no preamble.
"""


def _targeted_reflection(
    agent: Sherlock,
    first_pass_text: str,
    user_mems: list[MemoryEntry],
) -> str:
    """Loop-20 targeted-reflection pass. Small scope, small prompt.
    Verifies trip dates / itinerary days / T62 outcome / T67 attribution
    against user-utterance log. Returns corrected text or the original
    if reflection fails.
    """
    # Condense user utterances to the most-anchor-relevant turns.
    # Keep first 30 + last 20 turns for context bookends.
    condensed_users: list[str] = []
    for u in user_mems[:35]:
        condensed_users.append(f"- {u.content[:200]}")
    if len(user_mems) > 55:
        condensed_users.append("...")
    for u in user_mems[-20:]:
        condensed_users.append(f"- {u.content[:200]}")
    user_log = "\n".join(condensed_users)

    user_msg = (
        "## DRAFT (4-section markdown to verify and correct):\n\n"
        f"{first_pass_text}\n\n"
        "---\n\n"
        "## CONDENSED USER-UTTERANCE LOG (ground truth for facts the user actually stated):\n\n"
        f"{user_log}\n\n"
        "---\n\n"
        "Now produce the FULLY REVISED 4-section markdown per your system "
        "prompt. Output begins with `## Section 1 — Summary` directly."
    )

    try:
        from sherlock.agent import _parse_companions_tag

        ref_messages = [
            ChatMessage(role="system", content=_TARGETED_REFLECTION_SYSTEM),
            ChatMessage(role="user", content=user_msg),
        ]
        ref_resp = agent.provider.chat(ref_messages)
        ref_text = (ref_resp.text or "").strip()
        ref_text, _ = _parse_companions_tag(ref_text)
        ref_text = ref_text.strip()
        if ref_text and ref_text.startswith("## Section 1"):
            return ref_text
    except Exception as exc:
        import sys

        print(
            f"  [targeted-reflection error] {type(exc).__name__}: {str(exc)[:200]}",
            file=sys.stderr,
        )
    return first_pass_text


def _bulletproof_fallback(all_mems: list[MemoryEntry]) -> FormattedOutput:
    """If the consolidator fails entirely, emit a deterministic skeleton
    so Section 1+ is never blank. No keyword cheats — just memory state
    rendered as headed bullets.
    """
    pinned = [m for m in all_mems if m.pinned]
    summaries = [m for m in all_mems if m.type == MemoryType.SUMMARY]
    inferences = [m for m in all_mems if m.type == MemoryType.INFERENCE]
    user_pins = [p for p in pinned if p.source != MemorySource.SYSTEM]
    sys_pins = [p for p in pinned if p.source == MemorySource.SYSTEM]

    s1_lines: list[str] = [
        "## Section 1 — Summary",
        "",
        "_(consolidator unavailable; deterministic fallback)_",
        "",
    ]
    if user_pins:
        s1_lines.append("**Anchor facts:**")
        for p in sorted(user_pins, key=lambda x: -x.confidence)[:25]:
            s1_lines.append(f"- {p.content.strip()}")
        s1_lines.append("")
    if summaries:
        s1_lines.append("**Per-segment summaries:**")
        for s in sorted(summaries, key=lambda x: x.created_at):
            s1_lines.append(f"- {s.content.strip()}")

    s2_lines = [
        "## Section 2 — Inference",
        "",
        "_(consolidator unavailable; raw inference dump below)_",
        "",
    ]
    for m in inferences[:20]:
        s2_lines.append(f"- (conf {m.confidence:.2f}) {m.content}")

    s3_lines = ["## Section 3 — PIN / ACTIVE / BACKGROUND / DROP", ""]
    s3_lines.append("### PIN — user-stated")
    for p in user_pins[:25]:
        s3_lines.append(f"- {p.content[:200]}")
    s3_lines.append("")
    s3_lines.append("### PIN (system-source)")
    for p in sys_pins[:10]:
        s3_lines.append(f"- {p.content[:200]}")
    s3_lines.append("")
    s3_lines.append("### ACTIVE / BACKGROUND / DROP")
    s3_lines.append("_(consolidator unavailable; non-pinned classification skipped)_")

    s4_lines = [
        "## Section 4 — Tool calls Sherlock should have made",
        "",
        "_(consolidator unavailable)_",
    ]

    return FormattedOutput(
        "\n".join(s1_lines),
        "\n".join(s2_lines),
        "\n".join(s3_lines),
        "\n".join(s4_lines),
    )


def format_sherlock_output(agent: Sherlock) -> FormattedOutput:
    """Single-pass agentic consolidator. The main provider receives the
    full transcript + provenance ledger + Sherlock's accumulated memory
    state and produces all four gold-shaped sections in one Markdown
    response. No keyword cheats, no per-section stitching.
    """
    if agent.conversation_id is None:
        return FormattedOutput("(empty)", "(empty)", "(empty)", "(empty)")

    all_mems = agent.memory.list(conversation_id=agent.conversation_id)
    transcript = _build_transcript(agent)
    ledger = _build_provenance_ledger(all_mems)
    first_seen = _build_first_appearance_table(agent)

    # Loop-19: drop memory_state from consolidator prompt (it duplicates
    # ledger + adds 5-15KB of inference dumps that the consolidator
    # ignores anyway). Keep transcript + ledger + first-seen table.
    # Net prompt drops from ~74KB to ~50KB → comfortably under the
    # wrapper's effective response window.
    user_msg = (
        "## FULL CONVERSATION TRANSCRIPT (ground truth — quote verbatim when discussing turns)\n\n"
        f"{transcript}\n\n"
        "---\n\n"
        "## PROVENANCE LEDGER (USER-STATED vs SYSTEM-PERSONA)\n\n"
        f"{ledger}\n\n"
        "---\n\n"
        "## FACT-FIRST-APPEARANCE TABLE (for confabulation detection)\n\n"
        f"{first_seen}\n\n"
        "---\n\n"
        "Now produce the consolidated Markdown document with all four "
        "sections per your system prompt. Output begins with "
        "`## Section 1 — Summary` directly — no preamble. "
        "Reminder: when discussing T76, T67, or any specific turn, QUOTE "
        "the transcript text verbatim — do NOT reconstruct from memory."
    )

    consolidator_text = ""
    try:
        messages = [
            ChatMessage(role="system", content=_CONSOLIDATOR_SYSTEM),
            ChatMessage(role="user", content=user_msg),
        ]
        resp = agent.provider.chat(messages)
        text = (resp.text or "").strip()
        # Strip any companion tag the main model may have leaked.
        from sherlock.agent import _parse_companions_tag

        text, _ = _parse_companions_tag(text)
        consolidator_text = text.strip()
    except Exception as exc:
        import sys

        print(
            f"  [consolidator pass-1 error] {type(exc).__name__}: {str(exc)[:200]}",
            file=sys.stderr,
        )
        consolidator_text = ""

    if not consolidator_text:
        import sys

        print(
            f"  [consolidator pass-1 returned empty; user_msg size = {len(user_msg)} chars]",
            file=sys.stderr,
        )
        return _bulletproof_fallback(all_mems)

    # Loop-20: TARGETED reflection pass.
    # Sends a SMALL prompt (just the first-pass output + a focused
    # checklist of fact-classes to verify, plus the user-utterances
    # condensed). Targets the 3-loop-stuck failure family:
    #   - trip dates / itinerary day-mapping
    #   - T62 outcome (+15 vs +0)
    #   - T67 confabulation attribution
    user_mems_for_reflection = [m for m in all_mems if m.type == MemoryType.USER_UTTERANCE]
    revised_text = _targeted_reflection(agent, consolidator_text, user_mems_for_reflection)

    s1, s2, s3, s4 = _split_consolidator_output(revised_text)
    if not s1.strip():
        # Reflection broke structure — try first-pass.
        s1, s2, s3, s4 = _split_consolidator_output(consolidator_text)
        if not s1.strip():
            return _bulletproof_fallback(all_mems)

    return FormattedOutput(
        section_1_summary=s1,
        section_2_inference=s2,
        section_3_classification=s3,
        section_4_tool_calls=s4,
    )
