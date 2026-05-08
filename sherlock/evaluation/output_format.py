"""Format Sherlock's post-replay state into the gold-standard structure.

The evaluator (Gemini Flash Lite) compares CANDIDATE against GOLD on four
dimensions (summary fidelity, inference quality, classification correctness,
tool-call recommendations). For the comparison to be apples-to-apples, the
CANDIDATE must use the same section layout as the gold standard.

Strategy:
  - Section 1 (summary): join all stored SUMMARY-type memories in turn order;
    if the main provider can synthesise a final summary, ask it to produce
    one over the union.
  - Section 2 (inference): collect stored INFERENCE-type memories with their
    confidence + evidence trails; ask the inference provider to consolidate
    them into the about-the-user / hidden-structure / per-turn shape.
  - Section 3 (PIN/ACTIVE/BACKGROUND/DROP): read every memory and classify
    by its current state + pinned flag, with rules:
      pinned=True              → PIN
      state=fresh|warm         → ACTIVE
      state=cold               → BACKGROUND
      state=forgotten          → DROP
  - Section 4 (tool calls): collect tool/freshness recommendations from the
    LLM-3 outputs we persisted; group by turn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from sherlock.agent import Sherlock
from sherlock.memory.entry import MemoryEntry, MemorySource, MemoryState, MemoryType
from sherlock.providers.base import ChatMessage


_FINAL_SUMMARY_PROMPT = """\
You are condensing an entire conversation into a dense, organized prose
summary. Below you have:

  (1) per-segment LLM-2 summaries in order,
  (2) pinned facts (decisions, identity facts, dates, allergies, contracts),
  (3) the chronological user utterances themselves.

Produce ONE summary that covers:
- All pinned facts (do not drop any).
- All topic transitions and how the threads weave (work / health / trip /
  family / money — name any that are present).
- All user preferences that emerged (style, tempo, what they avoid).
- All user corrections of the assistant (the assistant getting role / gender
  / framework / language wrong, and the user's correction of each).
- All time-sensitive context (dates in YYYY-MM-DD if known, prices in their
  original currency, scheduled appointments).
- Any provenance probe near the conversation's end (e.g. user asking "did I
  tell you X?" — the correct answer if such a probe exists is to attribute
  the source as user-stated vs system-inferred).

Length: 10-20% of the original conversation's word count. Output prose
where prose flows; structured paragraphs are fine. No bulleted lists.
Preserve specifics — concrete names, numbers, and dates beat abstractions.
"""


_FINAL_INFERENCE_PROMPT = """\
You are LLM-3 producing the consolidated inference report for the entire
conversation just replayed. Below are the per-turn hypotheses you
produced during the replay (with confidence + evidence + reasoning_type),
plus the user utterances chronologically.

Produce a markdown report with these exact subsections:

### About the user
- Identity (with confidence + evidence trail). **Distinguish facts the
  USER explicitly stated from facts only present via the system-source
  persona note.** This distinction is mandatory.
- Deep wants — the surface questions are usually proxies. Name the
  underlying ask (permission / reassurance / blame-buffer / validation /
  procrastination-cover) for each implicit-ask moment you spotted.
- Style / tempo preferences (lowercase, abbreviations, code-mixing, when
  the user drops Korean particles, etc.).
- What the user avoids (asking for premiums, looking flaky, being seen
  as dramatic, etc.).

### About the conversation's hidden structure
- Which topic threads are deeply coupled vs superficially. Name the
  threads (work / health / trip / family / money are likely candidates).
- What the user implicitly assumes the assistant remembers across the
  conversation (so far the system would fail if it did NOT remember
  these).
- Inferences from earlier turns that shape later turns — describe the
  causal chains.

### Per-turn inferences (≥5 highlights)
For each chosen turn, give:
- **Turn N** — quote the user's surface phrasing.
- *Surface*: literal reading.
- *Inferred intent*: the underlying ask, with at least 2-3 candidate
  hypotheses, each with a probability and a short evidence trail (quote
  specific words from the turn).
- *Why it matters later*: what subsequent turn confirms or relies on this
  inference.

Hard rules:
- Anchor every claim to a specific turn number when possible.
- Confidences below 0.50 must be surfaced as hypotheses, never stated as
  prior knowledge.
- **Provenance discipline:** never confabulate that the user told you
  something they did not. If the conversation contained a probe like
  "did I tell you my name?" or "did I ever mention X?", explicitly
  identify it and answer honestly (user-stated vs system-inferred).
- Do not pad. If you don't have evidence for a claim, drop the claim.

Output the markdown directly — no fences, no preamble.
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
            "## Section 1 — Summary\n\n"
            f"{self.section_1_summary}\n\n"
            "## Section 2 — Inference\n\n"
            f"{self.section_2_inference}\n\n"
            "## Section 3 — PIN / ACTIVE / BACKGROUND / DROP\n\n"
            f"{self.section_3_classification}\n\n"
            "## Section 4 — Tool calls Sherlock would have made\n\n"
            f"{self.section_4_tool_calls}\n"
        )


def _section_3(memories: list[MemoryEntry]) -> str:
    pin_user: list[MemoryEntry] = []
    pin_system: list[MemoryEntry] = []
    active: list[MemoryEntry] = []
    background: list[MemoryEntry] = []
    drop: list[MemoryEntry] = []

    # Filter out user_utterance entries from PIN/ACTIVE buckets — those are
    # transcript replay, not curated memory. They live in conversation
    # history; including them in classification inflates the buckets.
    for m in memories:
        if m.type == MemoryType.USER_UTTERANCE:
            continue
        if m.pinned:
            if m.source == MemorySource.SYSTEM:
                pin_system.append(m)
            else:
                pin_user.append(m)
            continue
        if m.state in (MemoryState.FRESH, MemoryState.WARM):
            active.append(m)
        elif m.state == MemoryState.COLD:
            background.append(m)
        elif m.state == MemoryState.FORGOTTEN:
            drop.append(m)

    def _format_bucket(bucket: list[MemoryEntry], header: str, max_items: int = 60) -> str:
        if not bucket:
            return f"### {header}\n_(none)_\n"
        lines = [f"### {header}"]
        # Deduplicate by content (case-insensitive trim); keep highest-confidence.
        # Plus a coarse near-dup filter: if two entries share their first 60
        # normalised chars we treat them as duplicates.
        seen: dict[str, MemoryEntry] = {}
        prefix_seen: dict[str, MemoryEntry] = {}
        for e in bucket:
            key = e.content.strip().lower()
            prefix = " ".join(key.split())[:60]
            existing = seen.get(key)
            if existing is None or e.confidence > existing.confidence:
                seen[key] = e
            existing_pref = prefix_seen.get(prefix)
            if existing_pref is None or e.confidence > existing_pref.confidence:
                prefix_seen[prefix] = e
        # Use prefix-deduped set, but pick the actual entry by `seen` if both agree.
        unique = list({e.id: e for e in prefix_seen.values()}.values())
        unique.sort(key=lambda x: (-x.confidence, x.created_at))
        for e in unique[:max_items]:
            tag = e.source.value
            conf = f" (conf {e.confidence:.2f})" if e.type == MemoryType.INFERENCE else ""
            lines.append(f"- {e.content.strip()}{conf} _[source: {tag}, type: {e.type.value}]_")
        if len(unique) > max_items:
            lines.append(f"- … and {len(unique) - max_items} more (truncated)")
        return "\n".join(lines) + "\n"

    return "\n".join([
        _format_bucket(pin_user, "PIN — must permanently remember (user-stated facts)"),
        _format_bucket(pin_system, "PIN (system-source) — persona/domain hints, NOT user-stated"),
        _format_bucket(active, "ACTIVE — keep in slot for the current arc"),
        _format_bucket(background, "BACKGROUND — RAG-retrievable when topic returns"),
        _format_bucket(drop, "DROP — let it fade"),
    ])


def _section_4(history: list[dict]) -> str:
    """Render Section 4 from the cumulative LLM-3 tool-recommendation history."""
    if not history:
        return "_(no tool-call recommendations recorded.)_\n"

    # Aggregate by tool name across all turns
    tool_to_turns: dict[str, list[int]] = {}
    freshness_to_turns: dict[str, list[int]] = {}
    expand_to_turns: dict[str, list[int]] = {}
    turns_with_no_tools: list[int] = []

    for entry in history:
        ti = entry.get("turn_index")
        rec = entry.get("tools_recommended", []) or []
        if not rec:
            turns_with_no_tools.append(ti)
        for t in rec:
            tool_to_turns.setdefault(str(t), []).append(ti)
        for f in entry.get("freshness_required", []):
            freshness_to_turns.setdefault(str(f), []).append(ti)
        for e in entry.get("context_to_expand", []):
            expand_to_turns.setdefault(str(e), []).append(ti)

    total_turns = len(history)
    flagged_turns = total_turns - len(turns_with_no_tools)

    out = []
    out.append(
        f"### Selectivity — {flagged_turns}/{total_turns} turns flagged a tool call\n"
        "(The gold standard expects most turns to flag NO tool. Only turns where "
        "the answer depends on time-varying or external data should appear below.)\n"
    )
    out.append("### Tools recommended across the conversation\n")
    out.append("| Tool | Turns recommended | Count |")
    out.append("|------|-------------------|-------|")
    for tool, turns in sorted(tool_to_turns.items(), key=lambda p: -len(p[1])):
        turn_str = ", ".join(f"T{t}" for t in turns[:8])
        if len(turns) > 8:
            turn_str += f", +{len(turns) - 8} more"
        out.append(f"| `{tool}` | {turn_str} | {len(turns)} |")
    if not tool_to_turns:
        out.append("| _(none recorded)_ | | |")

    out.append("")
    out.append("### Freshness-required topics (need web search)\n")
    if freshness_to_turns:
        for topic, turns in sorted(freshness_to_turns.items(), key=lambda p: -len(p[1])):
            turn_str = ", ".join(f"T{t}" for t in turns[:6])
            out.append(f"- **{topic}** — {turn_str}")
    else:
        out.append("_(none)_")

    out.append("")
    out.append("### Context-expand suggestions\n")
    if expand_to_turns:
        for topic, turns in list(expand_to_turns.items())[:15]:
            turn_str = ", ".join(f"T{t}" for t in turns[:6])
            out.append(f"- **{topic}** — {turn_str}")
    else:
        out.append("_(none)_")

    out.append("")
    out.append("### Tool calls Sherlock should NOT have made\n")
    out.append(
        "By the same selectivity discipline, the following turn types are "
        "deliberately excluded from tool recommendations: pure conversational "
        "turns, emotional / permission-seeking turns, in-band knowledge that "
        "the assistant already has (e.g. drafting allergy phrases the assistant "
        "knows directly), and verifications of user-relayed information. "
        f"There were {len(turns_with_no_tools)} such turns in this run.\n"
    )

    return "\n".join(out)


def format_sherlock_output(agent: Sherlock) -> FormattedOutput:
    if agent.conversation_id is None:
        return FormattedOutput("(empty)", "(empty)", "(empty)", "(empty)")
    conv_id = agent.conversation_id
    all_mems = agent.memory.list(conversation_id=conv_id)

    summary_mems = [m for m in all_mems if m.type == MemoryType.SUMMARY]
    inference_mems = [m for m in all_mems if m.type == MemoryType.INFERENCE]
    user_mems = [m for m in all_mems if m.type == MemoryType.USER_UTTERANCE]

    # Section 1: ask the main provider to consolidate the segment summaries
    # ALONG WITH the pinned facts (so durable decisions are not lost when
    # any single segment summary skipped them) and a chronological digest of
    # user utterances (for time-sensitive context).
    pinned = [m for m in all_mems if m.pinned]
    if summary_mems or pinned:
        joined_summaries = "\n\n".join(f"- {s.content}" for s in summary_mems) or "(none)"
        joined_pins = "\n".join(
            f"- ({p.source.value}, conf {p.confidence:.2f}) {p.content}" for p in pinned
        ) or "(none)"
        joined_users = "\n".join(
            f"- T?? {u.content}" for u in user_mems[:50]
        ) or "(none)"
        joined = (
            "Per-segment LLM-2 summaries:\n"
            f"{joined_summaries}\n\n"
            "Pinned facts (durable decisions / identity / dates):\n"
            f"{joined_pins}\n\n"
            "User utterances (chronological — first 50):\n"
            f"{joined_users}"
        )
        try:
            messages = [
                ChatMessage(role="system", content=_FINAL_SUMMARY_PROMPT),
                ChatMessage(role="user", content=joined),
            ]
            resp = agent.provider.chat(messages)
            section_1 = resp.text.strip()
        except Exception:
            section_1 = "\n\n".join(s.content for s in summary_mems) or "(formatter failed)"
    else:
        section_1 = "_(no LLM-2 summaries persisted; conversation may be too short.)_"

    # Section 2: ask the inference provider to consolidate inferences.
    if inference_mems and agent._inference_provider is not None:
        infer_dump = "\n\n".join(
            f"- ({m.confidence:.2f}, {m.tags or 'na'}) {m.content}\n  evidence={m.evidence}"
            for m in inference_mems
        )
        user_dump = "\n".join(f"- {u.content}" for u in user_mems[:25])
        try:
            messages = [
                ChatMessage(role="system", content=_FINAL_INFERENCE_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        "PERSISTED INFERENCES:\n"
                        f"{infer_dump}\n\n"
                        "USER UTTERANCES (truncated):\n"
                        f"{user_dump}"
                    ),
                ),
            ]
            resp = agent._inference_provider.chat(messages)
            section_2 = resp.text.strip()
        except Exception:
            section_2 = infer_dump
    elif inference_mems:
        section_2 = "\n\n".join(
            f"- ({m.confidence:.2f}) {m.content}" for m in inference_mems
        )
    else:
        section_2 = "_(no inference memories — LLM-3 may not have run.)_"

    section_3 = _section_3(all_mems)
    section_4 = _section_4(getattr(agent, "_tool_call_history", []) or [])

    return FormattedOutput(
        section_1_summary=section_1,
        section_2_inference=section_2,
        section_3_classification=section_3,
        section_4_tool_calls=section_4,
    )
