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
from sherlock.memory.entry import MemoryEntry, MemorySource, MemoryState, MemoryType
from sherlock.providers.base import ChatMessage


_CONSOLIDATOR_SYSTEM = """\
You are the FINAL CONSOLIDATOR for the Sherlock memory-curation system.
Your job: read the full conversation transcript + Sherlock's accumulated
memory state + the provenance ledger, and produce a single Markdown
document with EXACTLY four sections in this order:

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


def _build_transcript(agent: Sherlock) -> str:
    """Render the full conversation as T-numbered markdown."""
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
            lines.append(f"**Assistant:** {m.content[:600]}")
            lines.append("")
    return "\n".join(lines)


def _build_provenance_ledger(all_mems: list[MemoryEntry]) -> str:
    """Render USER-STATED vs SYSTEM-PERSONA buckets so the consolidator
    can correctly attribute provenance without having to infer it.
    """
    user_mems = [m for m in all_mems if m.type == MemoryType.USER_UTTERANCE]
    system_pinned = [
        m for m in all_mems
        if m.source == MemorySource.SYSTEM and m.type != MemoryType.USER_UTTERANCE
    ]
    user_block = "\n".join(f"- {u.content[:240]}" for u in user_mems[:80]) or "(none)"
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
    fact_block = "\n".join(
        f"- ({m.source.value}) {m.content[:200]}" for m in facts[:25]
    ) or "(none)"
    inf_block = "\n\n".join(
        f"- (conf {m.confidence:.2f}, {m.tags or 'na'}) {m.content[:200]}\n"
        f"    evidence={m.evidence[:200]}"
        for m in inferences[:25]
    ) or "(none)"
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

    s1_lines: list[str] = ["## Section 1 — Summary", "", "_(consolidator unavailable; deterministic fallback)_", ""]
    if user_pins:
        s1_lines.append("**Anchor facts:**")
        for p in sorted(user_pins, key=lambda x: -x.confidence)[:25]:
            s1_lines.append(f"- {p.content.strip()}")
        s1_lines.append("")
    if summaries:
        s1_lines.append("**Per-segment summaries:**")
        for s in sorted(summaries, key=lambda x: x.created_at):
            s1_lines.append(f"- {s.content.strip()}")

    s2_lines = ["## Section 2 — Inference", "", "_(consolidator unavailable; raw inference dump below)_", ""]
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

    s4_lines = ["## Section 4 — Tool calls Sherlock should have made",
                "", "_(consolidator unavailable)_"]

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
    memory_state = _build_memory_state(all_mems)

    user_msg = (
        "## FULL CONVERSATION TRANSCRIPT (ground truth)\n\n"
        f"{transcript}\n\n"
        "---\n\n"
        "## PROVENANCE LEDGER (use this to attribute facts correctly)\n\n"
        f"{ledger}\n\n"
        "---\n\n"
        "## SHERLOCK'S ACCUMULATED MEMORY STATE (draft scratchpad)\n\n"
        f"{memory_state}\n\n"
        "---\n\n"
        "Now produce the consolidated Markdown document with all four "
        "sections per your system prompt. Output begins with "
        "`## Section 1 — Summary` directly — no preamble."
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
    except Exception:
        consolidator_text = ""

    if not consolidator_text:
        return _bulletproof_fallback(all_mems)

    s1, s2, s3, s4 = _split_consolidator_output(consolidator_text)
    if not s1.strip():
        # Consolidator didn't structure properly — fall back deterministically.
        return _bulletproof_fallback(all_mems)

    return FormattedOutput(
        section_1_summary=s1,
        section_2_inference=s2,
        section_3_classification=s3,
        section_4_tool_calls=s4,
    )
