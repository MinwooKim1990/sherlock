# AGENTS_AND_LOOP — How the coding agent should organize itself

> Version: v0.3 · 2026-05-08
> This file describes how the **coding agent that builds Sherlock** should structure its own work. It is not about Sherlock's runtime architecture — that is in `SPEC.md`.

---

## 0. Core stance

The user (Minwoo) wants minimal intervention. The coding agent is expected to:
- **Plan its own work**, including deciding what to parallelize.
- **Run a Ralph Wiggum loop** — keep trying, learn from failures, do not give up unless caps are hit.
- **Document its decisions** so the user can audit later without sitting through every loop.
- **Stop and call for help only when truly stuck** — see SOS protocol below.

The user will check in periodically. The agent should not block waiting for input unless explicitly required (Phase 1 / Phase 2 approvals in `EVALUATION_PROTOCOL.md`).

---

## 1. Orchestrator-and-subagents pattern

### 1.1 Who is the orchestrator

**The coding agent itself is the orchestrator.** When using Claude Code, this means the main Claude Code session is the orchestrator and uses the Task tool to spawn subagents. When using Codex CLI or another agent platform, use the equivalent capability.

The orchestrator's responsibilities:
1. Read this entire spec package on first run.
2. Build a dependency DAG of work to be done.
3. Identify the leaves of the DAG that can run in parallel.
4. Spawn subagents for parallel work; do sequential work itself.
5. Collect subagent results and merge.
6. Drive the Ralph loop.
7. Maintain logs.
8. Decide when to stop.

### 1.2 Who are the subagents

Subagents are spawned per task. The orchestrator decides task granularity — there is no fixed unit. Reasonable patterns:

- **Per-component within a milestone** — e.g. in M1, one subagent writes `providers/anthropic.py`, another writes `providers/openai.py`, both running concurrently.
- **Per-test-suite** — one subagent writes tests, another writes the implementation, they meet in the middle.
- **Per-feature in a vertical slice** — one subagent does memory CRUD end-to-end, another does CLI end-to-end.
- **Per-debugging-thread** — when an error appears, spawn a subagent dedicated to investigating it while the main thread continues elsewhere.

The orchestrator should bias toward parallelism when work is independent and toward sequencing when there are shared files or interface ambiguity.

### 1.3 Subagent communication protocol

Subagents **never** modify the spec files or `INTENT_DEVIATIONS.md`. They report back to the orchestrator. The orchestrator alone writes to logs and to spec.

A subagent task brief should include:
- The exact goal (one or two sentences)
- The file(s) it may write
- The file(s) it may only read
- The acceptance check (a test, or a manual verification step)
- The maximum time and tool-call budget
- The fallback if it cannot succeed (usually: report back, do not retry indefinitely; the orchestrator decides)

### 1.4 What runs in parallel vs sequentially

```
PARALLEL (good candidates)
  - Multiple independent provider implementations
  - Tests + implementation of independent modules
  - Documentation + final-stage refactoring
  - Investigation of a flaky test while main work continues

SEQUENTIAL (must serialize)
  - Anything that touches the same file
  - Steps that depend on the previous step's output (e.g. M2 needs M1's provider abstraction)
  - Phase 1 → Phase 2 → Phase 3 (dummy → gold → build)
  - Migration steps that change schema
```

When in doubt, sequential is safer. Parallelism is an optimization, not a requirement.

---

## 2. Ralph Wiggum loop

Ralph is the strategy: try the same thing differently, again and again, until it works. The loop is the heart of the build phase.

### 2.1 The loop, in pseudocode

```python
loop_id = 0
last_error_signature = None
same_error_count = 0
start_time = now()

while True:
    loop_id += 1
    log_raw({"event": "loop_start", "loop_id": loop_id, "elapsed": elapsed()})

    # 1. Read context: SPEC, last curated log, last test output, current state of code
    state = gather_state()

    # 2. Decide what to do next.
    #    Either: continue current milestone, retry failed step, fix specific error,
    #            or move to next milestone if exit criteria pass.
    plan = orchestrator_plan(state)
    log_curated_summary(plan)

    # 3. Execute the plan. May spawn subagents in parallel.
    result = execute(plan)
    log_raw(result)

    # 4. Run tests / evaluation.
    evaluation = run_tests_and_evaluation(state.current_milestone)
    log_raw(evaluation)

    # 5. Stop conditions.
    if evaluation.match_percent >= 80 and evaluation.all_milestone_exits_pass:
        log_curated("DONE — all milestones complete and dummy-conversation eval ≥ 80%.")
        write_DONE_marker()
        break

    # 6. Failure analysis: same error twice in a row?
    err_sig = signature_of(result.errors)
    if err_sig == last_error_signature:
        same_error_count += 1
    else:
        same_error_count = 1
        last_error_signature = err_sig

    # 7. SOS conditions.
    if same_error_count >= 5:
        write_SOS("Same error 5 times in a row. Stopping for human review.", details=result)
        break
    if elapsed() > config.time_cap_hours * 3600:
        write_SOS("Time cap exceeded. Stopping for human review.", details=summary())
        break
    if total_cost_usd() > config.cost_cap_usd:
        write_SOS("Cost cap exceeded. Stopping for human review.", details=cost_breakdown())
        break

    # 8. Otherwise: feed failure back as the next loop's input and continue.
    feedback = synthesize_feedback(result.errors, evaluation)
    log_curated(feedback)
    # loop continues
```

### 2.2 What "feed failure back" means

Each loop reads the previous loop's curated log. The curated log is **not** the raw log — it is the orchestrator's own summary of what happened and what to try next. Writing this curated summary at the end of each loop is mandatory. It is the agent's working memory.

A good curated entry contains:
- What was attempted this loop.
- What failed and why (root cause, not just stack trace).
- What will be different next loop.
- Open questions the agent has not resolved.

A bad curated entry just dumps stack traces. If the next loop cannot tell what to do differently from the curated log, the curated log is failing its job.

### 2.3 When to declare a milestone done

A milestone is done when **all of**:
1. Its Exit criteria in `SPEC.md` § 9 are met.
2. All tests for that milestone pass (the agent writes these per `EVALUATION_PROTOCOL.md`).
3. No regressions in earlier milestones (run their tests too).

The agent must not skip a milestone or fake an Exit criterion. If a criterion seems wrong or impossible, append to `INTENT_DEVIATIONS.md` with the reason and proposed replacement, and continue. The user will review later.

### 2.4 The 80% gate

The final, system-wide gate is in `EVALUATION_PROTOCOL.md`: Sherlock running on the dummy conversation must produce a summary+inference that scores ≥ 80% match against the gold-standard answer (judged by Gemini Flash Lite via the cli-wrapper-unified).

This gate is **not** per-milestone. It is the final "we are done" signal. The agent runs partial evaluations after M3 and beyond, but only the post-M9 score is the official one.

---

## 3. Guardrails

These are inviolable. Breaking them requires SOS.

### 3.1 Hard "do not"

- **Do not modify SPEC.md, README.md, or any spec file.** If you think the spec is wrong, append your reasoning to `INTENT_DEVIATIONS.md` and continue with your interpretation.
- **Do not skip Phase 1 or Phase 2 user approvals.** The gold standard is the only objective measurement in this project. Skipping it removes the success criterion.
- **Do not lower the 80% threshold to make the project pass.** If the system cannot reach 80%, that is information the user needs.
- **Do not delete logs.** Logs may be rotated or compressed but never destroyed.
- **Do not exceed cost or time caps without writing SOS.**
- **Do not silently swallow exceptions to make tests pass.** If a test passes only because errors are caught and ignored, the test is lying.
- **Do not commit secrets** (API keys, tokens) to the repo. Use env vars per the config schema.
- **Do not call the cli-wrapper-unified for purposes other than evaluation.** It is a dedicated evaluator, not a general LLM.

### 3.2 Soft "prefer"

- Prefer small, frequent commits over large monolithic ones. Each commit message should reference the milestone and what changed.
- Prefer pure functions and explicit dependencies. The system has many moving parts and untyped magic will accelerate the loop's failures.
- Prefer to write the test before the implementation when uncertain.
- Prefer to log over to print. Logs are the user's window.
- Prefer to ask once (via SOS) over to guess wrong silently.

### 3.3 Intent preservation

The user's intent is captured across:
- `SPEC.md` — what the system does
- This file — how the agent works
- `EVALUATION_PROTOCOL.md` — what success means
- `OPERATIONS.md` — how the environment is set up
- The conversation history that produced these documents (if available)

If the agent finds itself "fixing" by changing what the system does, that is drift. The agent should fix by changing how the code does the thing the system is supposed to do. Drift goes to `INTENT_DEVIATIONS.md`.

---

## 4. SOS protocol

### 4.1 When to write SOS

Write `logs/SOS.md` when any of:
- Same error signature appears in 5 consecutive loops.
- Time cap (config: `time_cap_hours`, default 12) exceeded without milestone progress.
- Cost cap (config: `cost_cap_usd`, default 50) exceeded.
- An external dependency is missing and cannot be installed (e.g. `cli-wrapper-unified` not on Desktop).
- The Phase 1 or Phase 2 self-assessment yields output the agent does not believe is top-tier (do not approve your own work below your standard).
- Any guardrail in §3.1 cannot be honored without breaking.

### 4.2 What SOS looks like

`logs/SOS.md` is a single file. Each SOS event appends a section:

```markdown
## SOS — 2026-05-08 14:32:11 — loop 47

**Reason:** Same error 5 times in a row.

**Error signature:**
```
ConnectionError: cli-wrapper-unified responded with EOF
```

**What I tried:**
- Reinstalling the wrapper from the Desktop path
- Downgrading the Gemini model id
- Increasing timeout to 60s

**What I think is wrong:**
The wrapper's binary on the Desktop appears to be a macOS-only build but the
sandbox is Linux. I cannot resolve this without a Linux build or running on
the user's Mac.

**What I need from the user:**
Confirm the cli-wrapper-unified location and platform compatibility, or
provide an alternative evaluation path.

**State:**
- Last successful milestone: M5
- Current attempt: post-M9 final eval
- Sherlock build: passing all milestone tests
- Gold standard: approved 2026-05-07

**Stopping now. Will resume on user signal (presence of file
`logs/SOS_RESOLVED.md` containing instructions).**
```

After writing SOS, the agent **stops the Ralph loop**. It does not retry. It waits for `logs/SOS_RESOLVED.md` to appear — the user will create this file with instructions, and the agent reads it and continues.

### 4.3 Asking for the user

The agent does not have notification channels (no Slack, email, etc., unless the user explicitly configures one). The SOS file appearing is the signal. The user will check periodically.

If the user has configured a notification webhook in `config.notifications.webhook_url`, the agent posts a one-line summary there too. This is optional and not required for v0.1.

---

## 5. The phases

Phase 0 through Phase 3 are described in `README.md`. Quick recap:

| Phase | Purpose | Blocking? |
|-------|---------|-----------|
| 0 | Sandbox bootstrap | No — runs to completion |
| 1 | Generate dummy conversation, self-judge top-tier, await user approval | **Yes** — must wait for user signal |
| 2 | Generate gold-standard answer, self-judge, await user approval | **Yes** |
| 3 | Build via Ralph loop, evaluate against gold standard | No — ends on DONE or SOS |

In Phases 1 and 2, the agent emits an SOS-like file (`logs/AWAITING_PHASE1_APPROVAL.md` and similar) and stops. The user creates an approval file (`evaluation/PHASE1_APPROVED` or `PHASE2_APPROVED`) when satisfied. The agent polls for these files (or is restarted by the user) and resumes.

---

## 6. Practical patterns

### 6.1 Test-first when fuzzy, code-first when concrete

If the agent is unsure how a function should behave, write the test first to force a decision. If the function is mechanical (a YAML loader, a sqlite query), code first and test second.

### 6.2 Subagent isolation

Subagents should not share mutable state with each other. They report back to the orchestrator. The orchestrator merges. This avoids race conditions on shared files.

### 6.3 Don't let the loop become busy-waiting

If the loop runs the same checks every minute with no work to do, that is a bug. The loop runs only after work has been attempted. Sleep between loops is an anti-pattern; if there is nothing to do, the loop ends or escalates to SOS.

### 6.4 Restart-resilience

Every loop must be safe to interrupt. Save state (current milestone, last error, version of companion prompts) to disk after each phase so a restarted agent can resume from the right place.

### 6.5 When to consult the user proactively

The agent should write SOS when stuck, but it should also write a lighter-weight `logs/CHECKPOINT.md` periodically (every milestone completion, or every 50 loops, whichever comes first). This file is short — 5 lines or so — describing what is done and what is next. The user reads CHECKPOINT.md to know if they need to intervene.

CHECKPOINT.md is informational only. It does not block the loop.

---

## 7. The agent's success criterion in one sentence

> **Build Sherlock per `SPEC.md`, until the dummy conversation processed by Sherlock yields a summary+inference that scores ≥ 80% match against the user-approved gold standard, with all milestone Exit criteria passing and no SOS unresolved.**

If you are reading this and the criterion is met, you are done. Write the final entry in `logs/curated.md`, write `logs/DONE.md` with a summary of what was built and final scores, and exit.
