# EVALUATION_PROTOCOL — How "done" is measured

> Version: v0.3 · 2026-05-08
> The single objective ground truth in this project. Without this file, "done" is undefined and the Ralph loop has no termination criterion.

---

## 0. The whole picture in one paragraph

Before building Sherlock, the agent generates a **long, realistic dummy conversation** that exercises the kinds of memory and inference patterns Sherlock is meant to handle. It then writes a **gold-standard summary + inference** for that conversation, representing the highest quality of what Sherlock should ultimately produce. Both artifacts are reviewed and approved by the user. Once approved, they become the evaluation target. The agent then builds Sherlock and, repeatedly, runs the dummy conversation through the implementation and compares the result against the gold standard using **`gemini-3.1-flash-lite-preview`** via the **`cli-wrapper-unified`** tool on the user's Desktop. When the match score reaches **≥ 80%**, the system passes.

---

## 1. Phase 1 — Dummy conversation generation

### 1.1 What the dummy conversation must contain

The dummy conversation is a multi-turn dialog between a hypothetical user and a hypothetical assistant. It must be **long** — at least 50 turns, ideally 80+ — and must exercise the following patterns intentionally:

- **Implicit references** ("that thing we talked about", "the same problem as before") that require Sherlock to recall earlier context.
- **Intent that is not literal**. The user says "should I buy this?" but means "tell me whether I will regret this." The assistant should infer the right thing.
- **Topic transitions** — the conversation moves between 3–5 distinct topics, each with its own threads of memory.
- **Time-sensitive facts** — at least one topic depends on knowing today's date and current information (e.g. an upcoming concert, a recent product release, a current pricing).
- **User preferences revealed gradually** — the user's tone, format preferences, and constraints should emerge over time, not all at once.
- **Corrections** — the user corrects an earlier assistant assumption at least twice. This tests whether Sherlock would learn from feedback.
- **Pinned facts** — at least three facts the user clearly wants permanently remembered (location, role, key dates).
- **Decay candidates** — at least three facts that come up once and never again, which a good Sherlock should let fade.
- **Tool affordances** — at least two moments where a tool call (web search, calculator) would obviously help.

### 1.2 Realism requirements

The conversation must read like an actual person talking to an actual assistant. Avoid:
- Robotic phrasing
- Unnaturally complete sentences from the user
- All-too-clean topic transitions
- The assistant being magically right every time

Embrace:
- Filler, abbreviation, casual style
- Mid-sentence topic shifts
- Half-finished thoughts the user expects the assistant to complete
- Occasional assistant mistakes (they are part of the test — Sherlock should remember the corrections)

The agent has freedom to invent the persona (a developer? a student? a parent?) but the persona should be coherent across the whole conversation.

### 1.3 Output format

Save to `evaluation/dummy_conversation.md` with the following structure:

```markdown
# Dummy Conversation for Sherlock Evaluation

## Persona summary
A 2-3 paragraph description of who the user is, their context,
and what they're trying to accomplish across the conversation.

## Conversation

### Turn 1
**User:** ...
**Assistant:** ...

### Turn 2
**User:** ...
**Assistant:** ...

...

### Turn N
**User:** ...
**Assistant:** ...

## Notes for evaluators
- Topics covered: ...
- Implicit references used: turn X referring to turn Y
- Pinned facts: ...
- Decay candidates: ...
```

### 1.4 Self-judgment

After writing the conversation, the agent **judges its own output**. The agent asks itself:

- Is this realistic enough that a real Sherlock could not tell it from a real conversation?
- Does it actually exercise the patterns listed in §1.1, or does it skim the surface?
- Is it long enough that a naive concatenated context would visibly hurt LLM 1's performance, making memory curation necessary?
- Are the implicit references genuinely implicit (the user does not re-state context)?

If the answer to all four is yes, mark it top-tier and proceed. **Otherwise rewrite.** Do not submit anything for user approval that the agent itself does not believe is top-tier.

The standard is: *"if I were a researcher publishing this as an evaluation benchmark, would I be embarrassed by it?"* If yes, it is not done.

### 1.5 Submitting for user approval

When the agent considers the dummy conversation top-tier:

1. Save final version to `evaluation/dummy_conversation.md`.
2. Write `logs/AWAITING_PHASE1_APPROVAL.md` with a short note: what makes this conversation good, what to look out for when reviewing, where the file is.
3. Stop the loop. Do not proceed to Phase 2.

The agent resumes when `evaluation/PHASE1_APPROVED` exists. The user creates this file with a single line — either `approved` or `revise: <reason>`. If revise, the agent reads the reason, rewrites, and resubmits.

---

## 2. Phase 2 — Gold-standard answer construction

### 2.1 What "answer" means here

The gold-standard answer is the **maximum-quality summary + inference** that Sherlock should ultimately be able to produce when given the dummy conversation as input. It represents what a perfect run of LLM 2 (summary) and LLM 3 (inference) would output.

It is **not**:
- A reply from the assistant in the dummy conversation
- A literal transcript

It **is**:
- A structured digest of the entire conversation, summarized with maximum fidelity
- Plus the inferences a Sherlock-style reasoner would draw, with hypotheses, evidence, and confidence

### 2.2 Required structure

Save to `evaluation/gold_standard.md`:

```markdown
# Gold Standard for Dummy Conversation

## Section 1 — Maximum-quality summary

A dense, organized summary of the conversation. Should preserve:
- All pinned facts
- All topic transitions
- All user preferences revealed
- All corrections the user made
- All time-sensitive context

The summary should be roughly 10-20% of the conversation's token count.
It should be prose where prose flows; structured where structure helps.

## Section 2 — Maximum-quality inference

For the conversation as a whole, what does a Sherlock-style reasoner conclude?

### About the user
- Who they are (inferred from clues, with evidence)
- What they want (deeply, not just surface-level)
- What they prefer (style, tempo, depth)
- What they avoid

### About the conversation's hidden structure
- What topics are deeply connected vs superficially
- What the user is implicitly assuming the assistant remembers
- What inferences from earlier turns should now influence later turns

### Per-turn inferences (selected highlights)
For at least 5 turns where inference is non-trivial, give:
- Turn N
- Surface meaning
- Inferred intent (with hypotheses, probabilities, evidence)
- Why this matters for future turns

## Section 3 — What Sherlock should remember vs forget

A list of every fact in the conversation, classified:
- PIN — must be permanently remembered
- ACTIVE — currently relevant, keep in slot
- BACKGROUND — RAG-retrievable when topic returns
- DROP — let it fade

## Section 4 — Tool calls Sherlock should have made

When in the conversation Sherlock would have benefited from:
- Web search (with the keywords)
- Date/time check
- Calculator
- File read
- Other
```

### 2.3 Self-judgment

Same as Phase 1 but stricter. The agent asks:

- Is this summary so good that re-reading it tells you everything important about the conversation?
- Are the inferences the kind a really thoughtful human reader would make, not generic "the user seems nice" platitudes?
- Are the evidence trails real — pointing to specific turns, specific phrases?
- Are the confidence numbers honest? (If something is genuinely 50/50, say 50%, not 80%.)
- Is the PIN/ACTIVE/BACKGROUND/DROP classification defensible?

If anything fails this bar, rewrite. The gold standard cannot be merely "good enough" — it is the ceiling Sherlock is being measured against. If it is mediocre, an 80% match is also mediocre.

### 2.4 Submitting for user approval

Same flow as Phase 1. Save to `evaluation/gold_standard.md`, write `logs/AWAITING_PHASE2_APPROVAL.md`, stop, wait for `evaluation/PHASE2_APPROVED`.

---

## 3. Phase 3 — Evaluation loop

After gold standard is approved, the build phase (Phase 3) begins. Periodically the agent runs an evaluation pass.

### 3.1 When to evaluate

- After M3 completes (first time inference is alive — sanity check, not the final score)
- After M5 completes (full async pipeline working)
- After M6 completes (evolution alive)
- After M9 completes — **this is the official run**

Earlier runs may score very low; that is expected. They exist to catch regressions, not to gate progress.

### 3.2 The evaluation procedure

```
1. Spin up a fresh Sherlock instance with a clean memory store
   and the test main_system_prompt.md (a generic helpful-assistant prompt
   the agent writes in M3 for testing purposes).

2. Replay the dummy conversation turn by turn. After each turn,
   let Sherlock's background pipeline complete (no skipping).

3. After the final turn, ask Sherlock to produce:
   a. Its full memory dump (all entries, with state, source, confidence)
   b. Its current active slot
   c. Its accumulated inferences from LLM 3

4. Format Sherlock's output to mirror the gold standard's structure
   (Section 1 summary, Section 2 inference, Section 3 classification,
    Section 4 tool calls).

5. Pass both gold standard and Sherlock's output to the evaluator.

6. Receive a match-percent score in [0, 100].

7. Log the score with timestamp, milestone, and full evaluator output.
```

### 3.3 The evaluator

The evaluator is **Gemini 3.1 Flash Lite Preview** (model id `gemini-3.1-flash-lite-preview` — verify exact id via the wrapper before running), called through the **cli-wrapper-unified** tool located on the user's Desktop.

**Locating the wrapper:**

The wrapper is at `~/Desktop/cli-wrapper-unified` or similar. The agent must verify the exact path during PHASE 0 and document it in `logs/curated.md`. If the wrapper is not found, the agent writes SOS — do not proceed without it.

**Calling the wrapper:**

The exact CLI invocation may vary. The agent should run `--help` first and document the actual interface. A reasonable call shape is:

```
<wrapper-cli> --model gemini-3.1-flash-lite-preview \
              --system <eval_system_prompt.txt> \
              --user-file <comparison_input.md> \
              --output <evaluator_output.json>
```

If the wrapper does not support `--model gemini-3.1-flash-lite-preview` directly, the agent finds the nearest valid id and notes the substitution in `INTENT_DEVIATIONS.md`.

### 3.4 Evaluator system prompt

The evaluator must score using a deterministic rubric. Save the system prompt to `evaluation/evaluator_system_prompt.txt`. A starting template:

```
You are an evaluation grader.

You will be given two documents:
1. GOLD — the human-approved reference answer
2. CANDIDATE — a system's attempt to produce the same answer

Both are summaries + inferences for the same source conversation.

Score the CANDIDATE against the GOLD on the following dimensions, each
in [0, 100]:

A. Summary fidelity (0-100):
   What fraction of the GOLD's facts and observations appear in the
   CANDIDATE, with correct emphasis? Penalize omissions and additions
   equally.

B. Inference quality (0-100):
   What fraction of the GOLD's inferences (intent hypotheses, evidence
   trails, classifications) are present in the CANDIDATE, with similar
   confidence and reasoning?

C. Classification correctness (0-100):
   For each fact, did the CANDIDATE classify it as PIN/ACTIVE/BACKGROUND/
   DROP the same way the GOLD did?

D. Tool-call recommendations (0-100):
   Did the CANDIDATE recommend the same tool calls at the same moments?

Final score = 0.4 * A + 0.4 * B + 0.1 * C + 0.1 * D.

Output JSON only:
{
  "summary_fidelity": <int>,
  "inference_quality": <int>,
  "classification_correctness": <int>,
  "tool_recommendations": <int>,
  "final_score": <int>,
  "notes": "<one paragraph explaining the score and the most impactful gaps>"
}
```

Tune as needed during M3 sanity runs. Document any tuning in `INTENT_DEVIATIONS.md`.

### 3.5 The 80% gate

When `final_score >= 80` from the official post-M9 run, the system is done.

Write `logs/DONE.md` with:
- Final score breakdown (all 4 dimensions)
- Path to the evaluator output JSON
- Path to Sherlock's output that scored
- Path to the gold standard
- Total loops, total time, total cost
- A short narrative of what worked and what did not

The agent then exits the loop.

### 3.6 What to do if score is below 80%

This is the Ralph loop's job. Read the evaluator's `notes` field — that is the diagnostic. Identify the worst dimension. Diagnose:

- Low **summary fidelity** → LLM 2 is dropping or distorting facts. Check summarization prompt, K-turn retention, decay timing.
- Low **inference quality** → LLM 3 is producing weak hypotheses or shallow evidence. Check Bootstrap output for LLM 3, evolution accumulation, confidence threshold.
- Low **classification correctness** → Decay engine is misclassifying. Check pin logic, semantic clustering.
- Low **tool recommendations** → LLM 3 is not surfacing tool needs. Check the `tools_recommended` field in its output schema.

Make a specific change, log it in `logs/curated.md`, run Phase 3 evaluation again. Do not change multiple things at once unless they are clearly related — that defeats the diagnostic.

### 3.7 What if the score plateaus

If after several Ralph iterations the score is stuck (improvements < 2%):

- Reread the gold standard. Is the agent's understanding of "good" misaligned?
- Is the gap structural (a feature of Sherlock that needs better design) or tuning (parameters)?
- Append findings to `INTENT_DEVIATIONS.md` if the gap suggests a spec change is needed.
- Write SOS if 5 consecutive iterations show no progress.

---

## 4. File layout for evaluation

```
evaluation/
├── dummy_conversation.md        # Phase 1 output, user-approved
├── gold_standard.md             # Phase 2 output, user-approved
├── evaluator_system_prompt.txt  # rubric for Gemini Flash Lite
├── PHASE1_APPROVED              # created by user (presence == approval)
├── PHASE2_APPROVED              # created by user
└── runs/
    ├── 2026-05-09T10-15-00/
    │   ├── sherlock_output.md   # what Sherlock produced this run
    │   ├── comparison_input.md  # GOLD + CANDIDATE concatenated for evaluator
    │   ├── evaluator_output.json
    │   └── score.txt            # final_score number for easy grep
    └── 2026-05-09T13-22-04/
        └── ...
```

Each run is timestamped. The agent does not delete old runs — they form the agent's progress trail.

---

## 5. The user's role in this protocol

The user does three things:

1. **Approve Phase 1.** Read `evaluation/dummy_conversation.md`, decide if it actually exercises the patterns. If yes, create `evaluation/PHASE1_APPROVED`. If not, create `evaluation/PHASE1_FEEDBACK.md` with specific issues.
2. **Approve Phase 2.** Read `evaluation/gold_standard.md`, decide if it represents a high enough ceiling. Same pattern.
3. **Resolve SOS when one appears.** Read `logs/SOS.md`, address the underlying issue, write `logs/SOS_RESOLVED.md` with instructions for the agent.

That is all. The user is not expected to read every loop's curated log, though they may.

---

## 6. Why this protocol matters

The hardest thing about Ralph loops is knowing when to stop. Without a fixed external standard, the agent can convince itself that any output is good enough. By committing to a user-approved gold standard up front and an external evaluator, the agent has no room to drift.

If the gold standard is wrong (too easy or too hard), the user can revise it and re-run. But the agent does not get to revise the gold standard.

This is why Phase 1 and Phase 2 cannot be skipped: they are the entire reason Phase 3 has a meaning.
