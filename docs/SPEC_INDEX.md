> **Note:** this is the original build-spec index used to construct the project.
> The product README (install, quick start, playground) now lives at the repo root: [README.md](../README.md).

# Project Sherlock

> A domain-agnostic context-curation library that lets a main LLM understand its own role and autonomously design how its companion reasoning models should think.
>
> Spec version: **v0.3** · Last updated: 2026-05-08

---

## What this directory is

This directory is a **complete specification package** for a coding agent (Codex CLI, Claude Code, or similar) to build Project Sherlock end-to-end with minimal human intervention.

The user (Minwoo) wrote his intent in conversation; the assistant compiled it into these files. The agent is expected to read all of them, build the system, and iterate via a Ralph Wiggum self-correcting loop until evaluation passes.

---

## Read in this order

| # | File | Purpose | Audience |
|---|------|---------|----------|
| 1 | `README.md` | This file. Navigation. | Everyone |
| 2 | `SPEC.md` | Core system specification. What Sherlock is and how it works. | Coding agent |
| 3 | `AGENTS_AND_LOOP.md` | How the coding agent should organize itself: orchestrator, subagents, Ralph loop, guardrails, SOS protocol. | Coding agent |
| 4 | `EVALUATION_PROTOCOL.md` | How "done" is measured: dummy conversation → gold-standard answer → Gemini-based scoring → 80% threshold. | Coding agent |
| 5 | `OPERATIONS.md` | Sandbox setup (lightweight venv, no Docker), logging schema (raw + curated), file layout. | Coding agent |
| 6 | `INTENT_DEVIATIONS.md` | Empty log. Coding agent appends here whenever it must deviate from spec, with reason. Reviewed by user. | Coding agent + user |

---

## The very first thing the coding agent does

After reading every file in this package, the agent runs the following sequence. **Do not skip phases.**

```
PHASE 0 — Sandbox bootstrap
  Set up venv per OPERATIONS.md.
  Initialize logging files per OPERATIONS.md.
  Verify the cli-wrapper-unified on Desktop is reachable; if not, log SOS.

PHASE 1 — Dummy conversation generation
  Generate a LONG synthetic multi-turn conversation per EVALUATION_PROTOCOL.md.
  Self-judge top-tier quality. Save to evaluation/dummy_conversation.md.
  Write SOS file requesting user review and approval. Stop and wait.

  (User reviews. If approved, the agent receives a signal file and proceeds.)

PHASE 2 — Gold-standard answer construction
  For the approved dummy conversation, produce the maximum-quality summary
  + maximum-quality inference per EVALUATION_PROTOCOL.md.
  Self-judge top-tier quality. Save to evaluation/gold_standard.md.
  Write SOS file requesting user review. Stop and wait.

  (User reviews. If approved, proceed.)

PHASE 3 — Build (Ralph loop driven)
  Implement Sherlock following SPEC.md, milestone by milestone.
  After each milestone, run the dummy conversation through the implementation
  and compute match-percent against the gold standard via the
  cli-wrapper-unified using gemini-3.1-flash-lite-preview.
  Loop: attempt → catch error → log → fix → retry.
  Exit when match-percent ≥ 80% and all milestone exit criteria pass.
  If stuck (same error N times, or time/cost cap), write SOS and stop.
```

---

## Core intent in one paragraph

Build a Python library called Sherlock. Users provide one thing: a main system prompt. The library bootstraps companion LLMs (a summarizer and an intent-inferrer) by having the main LLM design their system prompts itself. At runtime, those companions work asynchronously in the background to summarize history, predict where the conversation is going, prefetch relevant context (web search included), and fade memories that stop being useful. The user never has to repeat themselves. The system gets better at understanding this specific user as it runs. The main LLM should never have to ask "what did we talk about earlier" — Sherlock has already prepared the answer.

The success metric is concrete: produce a long synthetic conversation, write a gold-standard summary+inference for it, and Sherlock must reproduce that quality at 80%+ match.

---

## What the agent should NOT do

See `AGENTS_AND_LOOP.md` § Guardrails. Briefly:

- Do not modify `SPEC.md` or any spec file silently. If you believe the spec is wrong, append to `INTENT_DEVIATIONS.md` and continue with your interpretation.
- Do not skip Phase 1 or Phase 2 user approval. The gold standard is the only objective ground truth in this project.
- Do not attempt to bypass the 80% threshold by making the evaluator easier.
- Do not destroy logs. Logs are the user's window into your work.
- Do not exceed cost/time caps without writing SOS.

---

## Communication channels with the user

- **Logs** (`logs/raw.jsonl`, `logs/curated.md`) — the user reads these to understand current state.
- **SOS file** (`logs/SOS.md`) — when the agent stops or needs human input. The user is notified by file appearance.
- **Approval files** (`evaluation/PHASE1_APPROVED`, `evaluation/PHASE2_APPROVED`) — user creates these to signal go-ahead.
- **Intent deviations** (`INTENT_DEVIATIONS.md`) — append-only journal of changes from spec.

The user does not micromanage. The agent is expected to make decisions and document them.

---

## License & status

This is a working specification, not yet code. The first commit will be made by the coding agent during PHASE 0.
