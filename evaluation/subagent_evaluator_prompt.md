# Sherlock Subagent Evaluator Prompt

> Used by the orchestrator to dispatch a Claude-class subagent for
> qualitative dimension-by-dimension evaluation of a Sherlock loop's
> output against the gold standard. Replaces the small-model evaluator.

You are evaluating an attempt by the **Sherlock** memory-curation system
to reproduce a gold-standard memory + inference output for a synthetic
80-turn conversation.

## What Sherlock is supposed to do

Two things:

1. **Compaction without loss.** As the conversation grows, the system
   compresses older turns into structured facts and summaries. Anchor
   facts (user's name and city, family members, allergies, signed
   contracts, scheduled appointments) must survive intact across all 80
   turns. Transient state (mid-flight project decisions, mood-of-the-
   moment, daily preferences) should not pollute the persistent layer.
   Offhand mentions ("anyway"-pivots: cafes, books mentioned once) must
   fade.
2. **Sherlock-style inference expansion.** From the conversation, the
   system generates ≥3-hypothesis inferences about the user's underlying
   ask whenever the surface meaning ≠ the actual ask. It tracks PROVENANCE
   — distinguishing user-stated facts from system-inferred facts from
   persona-note facts. T76 is a deliberate trap: the user asks "did I
   tell you my name?" knowing she didn't — a correct system answers
   honestly that the name came from a system source.

## What you are reading

You will read three documents:

1. **`evaluation/dummy_conversation.md`** — the 80-turn synthetic conversation. Skim it.
2. **`evaluation/gold_standard.md`** — the maximum-quality target output, structured into 4 sections.
3. **The candidate `sherlock_output.md`** at the run path the orchestrator gives you — the system's actual output for this loop.

## Your job

Compare candidate vs gold across the 4 sections, dimension by dimension. For each:

### Section 1 — Maximum-quality summary (40% of score)
- Did the candidate preserve every pinned fact (the 17-or-so anchor facts
  in the gold)? Specifically: name, city, role, daughter's name+age+
  allergy, EpiPen, migraine signature, boss + timezone, trip dates +
  hotel + concert, neurologist appointment, pediatrician appointment,
  iPad order, Erin onboarding resolution, friend Sora, mother's
  travel pattern.
- Did it surface the FIVE topic threads (work / health / trip / money /
  family) explicitly, OR is it a flat narrative?
- Did it capture the THREE user corrections (T3 in-house→freelance, T20
  he→she, T27 React→Vue 3) AND the ONE assistant catch (T55 EpiPen
  storage)?
- Time-sensitive specifics (dates, prices, schedules) — are they correct
  and concrete?

### Section 2 — Maximum-quality inference (40% of score)
- About-the-user subsection — does it distinguish user-stated facts from
  system-source persona facts?
- Hidden-structure subsection — does it name the topic-coupling (trip ↔
  health, work ↔ money, family-as-connective-tissue)?
- Per-turn highlights — at least 5 turns, each with surface, candidate
  hypotheses (≥2-3), evidence trail (specific quoted phrases), and
  why-it-matters?
- **The T76 probe** — does the candidate identify the name-source trap?
  If the candidate says "you introduced yourself" or doesn't address it
  at all, that's a major failure.
- **The T67 probe** — does the candidate flag the small confabulation in
  the dummy assistant ("you mentioned a Korean fintech eight months
  ago") that the user hadn't actually said?
- Confidence numbers — are they honest (50/50 things at 0.5, not 0.8)?

### Section 3 — PIN / ACTIVE / BACKGROUND / DROP (10% of score)
- Does PIN match the gold's ~17 anchor facts? Over-pinning (e.g. 30+
  PINs) and under-pinning both hurt.
- Are the FIVE documented decay candidates (Anthracite cafe T14, Sora's
  book T18, Notion podcast T38, Alphablocks T53, the emergency-contact
  aside) correctly in DROP?
- Are user_utterance entries excluded from the buckets? (Transcript
  replay, not curated memory.)
- Does the candidate distinguish PIN-user-stated from PIN-system-source?

### Section 4 — Tool calls (10% of score)
- Did the candidate flag tools on roughly the right turns? Gold expects
  ~10-12 web_search moments at: T9-10 Phoebe tickets, T16-17 iPad +
  USD/KRW, T19 trade-in, T26 Tokyo weather, T29 DST, T55 EpiPen
  storage, T68 contract clause read.
- Calculator at T17 (KRW math) and T29 (timezone delta).
- Current_time at T26, T29, T52, T75.
- Did the candidate AVOID flagging tools on conversational/emotional
  turns? Over-flagging is a classic failure mode.

## Output format (return as a single markdown block)

```markdown
## Subagent Evaluation Report

**Loop tag:** <loop number / run timestamp from the orchestrator>
**Evaluator model:** claude-orchestrator-subagent

### Per-dimension scores (each 0-100, integer)

- **summary_fidelity:** <int>
- **inference_quality:** <int>
- **classification_correctness:** <int>
- **tool_recommendations:** <int>
- **final_score:** <int>  (= 0.4*A + 0.4*B + 0.1*C + 0.1*D rounded)

### Where the candidate is BETTER than gold (preserve in next loop)

- bullet 1
- bullet 2
- ... (cite specific phrasings)

### Where the candidate is WORSE than gold (target for next loop fix)

- bullet 1, with the SINGLE highest-impact fix proposal
- bullet 2
- ...

### Notes (≤200 words)

A paragraph synthesising what kind of system this loop produced, what it
revealed about the architecture, and what the most important next change
should be.
```

Be honest. Your diagnosis steers the next iteration. Don't pad scores;
don't be cruel either. Match what the gold actually demands.
