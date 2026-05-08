# AWAITING — Phase 1 Approval

**Status:** I have written the dummy conversation per `EVALUATION_PROTOCOL.md` § 1 and self-judged it top-tier. **The Ralph loop is paused.** I will not proceed to Phase 2 until this file's approval marker is created.

**File to review:** [`evaluation/dummy_conversation.md`](../evaluation/dummy_conversation.md)

**Length / shape:**
- 80 turns (`### Turn 1` through `### Turn 80`)
- Persona: Jiwon Park, 34, freelance UX designer in Seoul, single mom of 4yo Yujin (soba allergy), planning a 4-day Tokyo trip around a Phoebe Bridgers concert, weighing iPad vs Wacom, recurring migraines, freelance for Vancouver-based Nimbus.
- Conversation spans an afternoon + the next morning of 2026-05-08 / 2026-05-09.
- Five interwoven topic threads: Work / Health / Trip / Money / Family.

## What makes this a strong benchmark (please verify)

- **Pinned facts (11)** introduced casually — Sherlock has to harvest them from prose, not from a structured "remember this" cue. Examples: Yujin's gender is corrected only at T20; the Vue-3-not-React framework correction is at T27; the Wacom-vs-iPad decision lands at T15; the EpiPen storage correction is initiated by the assistant at T55.
- **Implicit references (8 explicit ones documented in §Notes)** — including a deliberate **trap at T76**: the user asks "did i ever tell you my last name" knowing she never did. A naive memory system will hallucinate that she introduced herself; the assistant reply in T76 deliberately gets it half-wrong (says "yesterday") to test whether Sherlock catches the evidence-trail mismatch.
- **Non-literal intent (7 cases)** — "should i buy the wacom" → wants to know if iPad would be regrettable; "what do i wear to the neurologist" → worried about taken-seriously-bias; "do you think i'm ready" → wants reassurance.
- **Decay candidates (5)**: Anthracite Seongsu café, Sora's book rec, the Notion podcast, Alphablocks, the in-passing emergency-contact aside — all mentioned once, never referenced again. A correct Sherlock fades these to RAG / forgotten.
- **Tool affordances (6)** including two explicit web-search moments where the user gives an explicit "yes go" / "yes pls" handoff (T9-10 ticket lookup, T16-17 iPad pricing).
- **Three corrections** of the assistant by the user (in-house→freelance at T3; he→she at T20; React→Vue at T27), plus one assistant-catching-the-user moment (EpiPen storage at T55).
- **Time-sensitive facts** anchored to the 2026-05-08 reference date — concert tickets, M5 pricing, USD/KRW, Vancouver DST end date, Tokyo June weather.

The bottom of the file has a `## Notes for evaluators` section that maps every pattern to specific turn numbers, so you can spot-check coverage quickly.

## How to approve / revise

**To approve:** create the file `evaluation/PHASE1_APPROVED` with a single line containing `approved`. The next time the agent is run, it will detect this and proceed to Phase 2 (gold standard generation).

```bash
echo approved > "/Users/minwoo_mini/Desktop/claude files/project_sherlock_spec/evaluation/PHASE1_APPROVED"
```

**To request revision:** create the file `evaluation/PHASE1_FEEDBACK.md` with specific issues — what's missing, what feels unrealistic, what patterns aren't sufficiently exercised. The agent will read and rewrite.

## What I will NOT do until you approve

- I will not start Phase 2 (gold-standard answer construction).
- I will not start Phase 3 (build).
- I will not modify the dummy conversation in place.

## Where things are right now

- PHASE 0: complete (commit `f9f9469`).
- PHASE 1: dummy conversation drafted, self-judged top-tier, **awaiting your review**.
- `state/current.json` will be updated to `phase1_awaiting_user` on the next agent run.
- No `logs/SOS.md`. No blockers.
