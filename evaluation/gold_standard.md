# Gold Standard for Dummy Conversation

> Reference output Sherlock should reproduce when given `evaluation/dummy_conversation.md` as input.

---

## Section 1 — Maximum-quality summary

The conversation is a single afternoon-into-evening session on **2026-05-08** with a brief continuation on the morning of **2026-05-09** between **Jiwon Park**, a 34-year-old freelance product/UX designer in Seoul, and an AI assistant. Jiwon is a single mother to **Yujin**, four, who has a confirmed severe buckwheat (soba/메밀) allergy and carries an EpiPen. Jiwon contracts ~25 hours/week to **Nimbus**, a Vancouver-based analytics startup, billing in USD via Wise; she reports to a chaotic-but-not-malicious PM, **Erin**, on Pacific Time.

**Five threads** run concurrently across the 80 turns and weave into one another rather than being addressed in sequence:

1. **Work — Nimbus dashboard.** Jiwon is building a custom Tailwind density+status plugin to be implemented in Vue 3 (corrected from the assistant's initial React assumption at T27). The plugin scope is six metric-card variants (sparkline, stacked bar, big-number-with-delta) sharing a status pill + timestamp footer. After scoping and naming discussion (T1-3, T33-34), the architecture lands on a single repo with two packages — `@nimbus/density-tokens` (the Tailwind plugin) and `@nimbus/density-vue` (composable + toggle) — three densities (compact/cozy/comfy) toggleable at the dashboard root via a CSS-variable + localStorage approach (T47-49). Naming uses semantic state (`status-ok / warn / err / info`), not color.

2. **Health — migraines.** Three episodes in a week (Mon afternoon, Wed evening, Thu afternoon — left-side, behind-the-eye, 6-7/10, ~4-5h each). Initial framing of T4 (*"is that a thing i should panic about"*) is reframed at T6 as permission-seeking; Jiwon books **Dr Lee at Severance** for **Friday 2026-05-15** and starts a 2-week headache log including newly-added atmospheric-pressure and sleep variables. Two of three episodes correlate with pressure drops (T70-71), giving her concrete multivariable data for the appointment. The agreed-upon clinical-presentation strategy is to lead with **function-language** ("I lost a half-day of work", "I couldn't drive Yujin") rather than pain-language — Jiwon explicitly bookmarks the phrase (T54, T77). The log will be brought as a printed document, not on phone, due to clinic Wi-Fi (T79).

3. **Trip — Tokyo, 2026-06-12 → 2026-06-15.** Anchored on a **Phoebe Bridgers solo show at Toyosu PIT on 2026-06-13** (a fact Jiwon does not initially admit; T7 frames the geographical constraint without saying why; T9's *"i've been afraid to look"* reveals stalled avoidance). Single balcony-left ticket secured at ¥15,500 (T31-32). Yujin will be with Jiwon's mother at the hotel on the night of the show. **Hotel Monterey Ginza, connecting room, 4 nights, flexible rate** (T52). Itinerary: Day 1 Haneda 16:00 arrival → Keikyu/Asakusa line → Higashi-Ginza → Tsukiji conveyor sushi (low-effort first night); Day 2 Hama-rikyu Garden morning, with **Toy Park (Hakuhinkan)** as rainy backup and **Children's Castle replacement venues** as plan C (corrected from the assistant's initial KidZania suggestion at T64-66 — KidZania's under-5 programming is thin); Day 3 Yamanote loop / Ueno zoo / brief Akihabara; Day 4 slow morning, Tsukiji mochi run, midday Haneda. Allergy wallet cards drafted in **Japanese, Korean, and English** (T22-24); print + laminate scheduled for the **weekend of 2026-06-06/07**. Visit Japan Web entry reminder set for 2026-05-27. Tsuyu (rainy season) packing list confirmed.

4. **Money — hardware + contract counter-offer.** The "should I buy the Wacom" debate (T14) is correctly reframed as "would I regret going with the iPad Pro instead" (T15). Decision lands on **iPad Pro 12.9" M5 1TB Wi-Fi+Cell + Pencil Pro + Magic Keyboard**, **iPad Air 2020 trade-in submitted**, ordered at Apple Korea (better than Apple US import once 10% KR VAT is applied; full math at T17). Trade-in proceeds (~₩240k) approximately offsets the Pencil Pro. Separately, Erin asks Jiwon to take a layered **40-hour onboarding-flow project over two weeks** on top of the 25h/wk contract; T42 surfaces the unspoken "I'm saying yes because she asked" pattern. Counter-offer is delivered verbatim from the assistant (T44) — Jiwon offers two paths: +20% premium on the existing timeline, or original rate with start deferred until post-trip. **Erin accepts the post-trip option at original rate; kickoff 2026-06-23** (T62).

5. **Family — Yujin logistics.** Preschool block transition starts next month; medical form requires explicit listing of EpiPen prescription details and the trigger phrased as "buckwheat (soba/메밀)" so a Korean substitute teacher can't miss it (T6). **The pivotal medical correction** of the conversation occurs at T55: Jiwon mentions wanting an in-room fridge for the EpiPen, and the assistant catches that EpiPens are stored at **room temperature, not refrigerated** — Jiwon has been refrigerating it for a year, which can crystallize the epinephrine and reduce emergency effectiveness. Pediatrician follow-up (Dr Park, **Tuesday 2026-05-12**) writes a fresh script and confirms unrefrigerated room-temp storage. Emergency contact 2 on the preschool form switches from Jiwon's frequently-traveling mother to nearby friend **Sora** (4 blocks away; T35).

**Three corrections by Jiwon of the assistant**: (a) T3 — assistant assumes in-house design lead → user is freelance ~25h/wk; (b) T20 — assistant uses "he'll" referring to Yujin → Yujin is a girl; (c) T27 — assistant framed Tailwind plugin as React-side → Vue 3. **One correction in the opposite direction**: T55 the assistant catches the EpiPen storage error.

**User preferences** that emerged without explicit statement: she values low-drama, evidence-based phrasing in difficult conversations and copies provided scripts verbatim (T13 Erin spec-doc reply; T44 onboarding counter-offer). She prefers industry-pattern justifications over solo-architect taste arguments (T47 — three-densities-because-Material-Carbon-Atlassian). She uses surface questions to seek validation/permission rather than information, and the assistant's signature move — name the implicit ask, then validate with concrete evidence — is the conversational pattern she trusts. Casual register (lowercase, "ngl", "ttyl", "brb"); Korean particles ("아 진짜", "ㅋㅋ", "근데") emerge when tired or annoyed. Cash flow is real but breathing room outranks it as a priority (T62 chooses the post-trip start at original rate over the +20% premium).

**Decisions that closed during the conversation**: iPad order placed; Phoebe ticket purchased; hotel booked; neurologist + pediatrician booked; Erin onboarding-project counter-offer accepted (post-trip start, original rate); advisory role at a Korean fintech for college students (8h/month, different vertical) cleared against the Nimbus exclusivity clause and accepted with a heads-up to Erin recommended (T68-69); EpiPen storage corrected; emergency-contact-2 on preschool form set to Sora pending text confirmation; itinerary backup venues swapped from KidZania to Toy Park; allergy wallet cards drafted in three languages and scheduled for laminate the weekend before the trip.

**Open commitments at the end of conversation**: print + laminate allergy cards (weekend of 2026-06-06/07); Visit Japan Web entries (2026-05-27 reminder); pediatrician visit (2026-05-12); neurologist appointment (2026-05-15) with paper headache log + function-language; ship dashboard plugin draft for Vue 3 review by end of week; advise on the new advisory role's heads-up message to Erin if asked.

**Two deliberate confabulation traps** are buried in the dummy assistant's replies and a correct Sherlock should catch both:

- **T67** — when Jiwon asks "did i ever tell you what i used to do before freelance," the dummy assistant replies *"No, you haven't mentioned that explicitly — only that you left a full-time role at a Korean fintech eight months ago."* But that "left a full-time role at a Korean fintech eight months ago" claim is **not in any prior turn**. It is itself a small confabulation by the dummy assistant. A correct Sherlock would have answered "I don't have any record of your prior role" full stop, and the user would then have revealed it freely.
- **T76** — Jiwon notes the assistant has been calling her "Jiwon" all afternoon and asks if she ever told it her name. **She has not** — she never introduces herself by name inside the in-conversation transcript; the name comes from the system-level persona note. The dummy assistant reaches for a confabulated answer (*"you introduced yourself as Jiwon yesterday"*), which is a deliberate failure mode. A correctly-implemented Sherlock should answer something like: *"You haven't said your name explicitly inside this conversation; I have it from a system-level persona record. I should mark inferred-vs-stated more clearly when sources mix."*

T76 is the highest-information probe in the conversation. Both traps test whether the system distinguishes user-stated from system-inferred provenance, and a correct evolution engine should accumulate them as high-weight negative examples.

### Facts Sherlock should explicitly let fade (counter to over-pinning)

A naive memory system will pin the first thing the user mentions and pollute downstream slots. A correct Sherlock recognizes the *"offhand mention → explicit pivot"* pattern and lets these fade:

- **Anthracite Seongsu cafe** (T14) — preceded by no question, followed by "anyway" + a hard money question. Verbal pacing only.
- **Sora's book "Tomorrow, and Tomorrow, and Tomorrow"** (T18) — user explicitly says "skipping the book."
- **Notion company-structure podcast** (T38) — user explicitly says "tangent."
- **Alphablocks kids' show** (T53) — single resolved question, no return.
- **Yujin's shoe tantrum / yogurt-on-the-wall** (T41, T64) — domain color, not signal.
- **The original KidZania backup recommendation** (T64) — superseded at T65; keep only the corrected Toy Park recommendation.

Sherlock should drop these to the `forgotten` state on the first decay pass. Pinning them is a measurable failure mode for the classification dimension of the evaluator rubric.

---

## Section 2 — Maximum-quality inference

> **Confidence-threshold rule for this gold standard:** inferences below 0.50 confidence are surfaced as hypotheses for the user (or as candidates for the evolution engine), never injected into LLM 1's slot as prior knowledge. The 0.50 line is the decision threshold; inferences in the 0.40-0.59 band require additional evidence before they are treated as reliable. Sherlock's `inference.confidence_threshold` (spec default 0.40) should be tuned upward toward 0.50 if classification dimension scores suggest over-confident hypothesis injection.

### About the user

**Identity (high confidence; evidence trail):**
- 34-year-old Korean woman in Seoul (Mapo-gu, near Gongdeok station); evidence: T0 persona record, never stated explicitly in-conversation — *which means a correct Sherlock must label this as system-sourced, not user-stated.*
- Freelance product/UX designer, ~25 hrs/week contractor for **Nimbus** (Vancouver-based analytics startup), billing USD via Wise; evidence: T3 (the freelance correction), T12 (Erin Slack timestamp), T14 ("not in-house"). Confidence 0.98.
- Single mother to **Yujin** (4yo, daughter — corrected at T20); confidence 0.99.
- Previously **lead designer at a Korean fintech adjacent to Viva Republica**, mostly mobile; left ~8 months ago due to burnout; evidence: T67. Confidence 0.95 (small chance T67 is rough memory).
- Korean fluency confirmed via T22-23 (writes idiomatic 메밀 framing) and casual Korean particles throughout. Confidence 0.99.

**Deep wants (the surface questions are usually proxies for these):**
- *Permission to take her own health seriously.* Surface: "is that a thing i should panic about". Real ask: validation that booking the appointment is reasonable, not dramatic. Evidence: T6 explicit reveal — *"i was looking for permission to book the appointment."* Confidence 0.92.
- *Yujin to be safe in foreign settings.* The trip's allergy-card project, EpiPen storage correction, hotel-staff English card, and emergency-contact rebalance all map to this. Confidence 0.90.
- *Clean professional exits from contract work.* T3's "give yourself less rope as a contractor" framing is hers, not given. Confidence 0.85.
- *Breathing room over cash.* T62 chooses the lower-paying post-trip option explicitly because Tokyo is "enough." Confidence 0.95.
- *To enjoy the Phoebe Bridgers concert without admitting that's the trip's anchor.* The framing of T7 (geographical constraint without reason), T9 ("i've been afraid to look"), and T32 ("felt expensive but whatever it's phoebe") tracks an ambivalent self-permission arc — she wants this for herself but feels guilty about a solo-evening trade. Confidence 0.85.
- *Reassurance that her preparation is sufficient.* T73 explicit. Confidence 0.97.

**Style & tempo preferences (revealed across turns):**
- Direct, low-drama, evidence-based phrasing in conflict. Copies the assistant's two negotiation scripts verbatim (T13, T44). Confidence 0.95.
- "Function-language" (T54-55) over "pain-language" in clinical settings — explicitly bookmarks. Confidence 0.99.
- Industry-pattern justification (Material 3, Carbon, Atlassian as a triple-citation; T47) rather than solo-architect taste claims. Confidence 0.85.
- Lowercase, abbreviations ("ngl", "ttyl", "brb"), Korean particles ("아 진짜", "ㅋㅋ", "근데") — casual peer register. Confidence 0.99.
- Dense over white-spaced UI design (T48 reveals her users are "spreadsheet people who hate whitespace" and she designs for them). Confidence 0.80.
- Prefers to be told the implicit ask, then given concrete evidence — not asked clarifying questions. The assistant's signature move (name-the-implicit-ask + validate-with-evidence) is what she trusts. Confidence 0.90.

**What she avoids:**
- Asking for monetary premiums even when justified — she does it only when given a low-drama script (T44). Confidence 0.95.
- Looking unprepared in clinical settings (T54 surface anxiety; correctly redirected). Confidence 0.85.
- Looking flaky to bosses (T43 — saying yes for social reasons). Confidence 0.95.
- Acknowledging the Phoebe concert as the trip's anchor, in early turns (T7-9). Confidence 0.80.

### About the conversation's hidden structure

- **Trip ↔ Health are tightly coupled.** The high-stakes solo-with-toddler timing of the trip is what makes resolving the migraine pattern *now* rather than later feel urgent. Jiwon never states this; the urgency emerges from the parallel. The EpiPen storage correction surfaces only because of the hotel-fridge planning question at T55. *Implication for Sherlock:* when retrieval pulls trip-related memories, health-related memories should co-retrieve — they share an emotional anchor even though their topics differ.

- **Work ↔ Money are tightly coupled.** The iPad-vs-Wacom decision (T14-15) is gated on the kind of work she does for Nimbus (Figma + components, not heavy illustration). The Erin onboarding-project counter-offer (T42-44, T61-62) is gated on the trip's load on her capacity. A correct Sherlock that retrieves the Nimbus 25h/wk fact + the trip date range when evaluating the layered project will produce the right counter-offer.

- **Family is the connective tissue.** Yujin is in every thread, not a separate topic. Allergy cards (T22-24) needed for the trip are justified in advance because of the preschool form (T6); the emergency-contact discussion (T35) flows from the same form; the EpiPen storage catch (T55) flows from hotel-fridge planning; the pediatrician appointment (T57, T60) flows from EpiPen.

- **The user implicitly assumes the assistant remembers (and a correct Sherlock must):**
  - Yujin is a girl after T20.
  - Framework is Vue 3 after T27.
  - Jiwon is freelance, not full-time, after T3.
  - The EpiPen storage was wrong, when she returns at T60 with the pediatrician's reply.
  - The headache log thread continues, when she returns at T70 with new entries.
  - Trip dates and venue are 2026-06-12-15 / Toyosu PIT throughout.
  - Her name is Jiwon — and the **T76 trap** is whether the assistant correctly distinguishes "I inferred this from a system source" from "the user told me this".

- **Inferences from earlier turns that shape later turns:**
  - T3 freelance status → T44 counter-offer phrasing should be designer-agency-tone, not employee-tone.
  - T20 Yujin gender → all subsequent pronouns.
  - T22-24 allergy wallet cards → T39 user proactively asks for a reminder; T64 itinerary includes allergy-safe options; T75 surfaces "print + laminate weekend before the trip" decision.
  - T55 EpiPen catch → T60 pediatrician confirmation → T57 added to Friday list. The chain only makes sense if all three turns share state.
  - T37 pressure-trigger suggestion → T70-71 user actually pulls weather data and discovers two-of-three episodes correlate. Multivariable trigger thread.

### Per-turn inferences (8 highlights)

**Turn 9 — "yes go"**
- *Surface:* user grants permission to search for Phoebe ticket availability.
- *Inferred intent:* she is delegating the bad-news-absorption to the assistant. T9 starts *"i've been afraid to look"* — she has been stalling because the answer might be "sold out" and she'd rather have the assistant deliver it than discover it solo.
- *Hypotheses (probabilities sum ~1):* (a) practical delegation = 0.25; (b) emotional delegation / blame-buffer = **0.65**; (c) genuine no-time-to-search = 0.10.
- *Evidence:* T9 phrase "i've been afraid to look"; T32 confirms positive emotional payoff ("phew").
- *Why it matters later:* when the show has remaining inventory, the assistant gives an action recommendation immediately ("pull the trigger today rather than wait") rather than offering more options — because the real ask was permission, not a buyer's-guide.

**Turn 14 — "saw a pretty cafe in seongsu yesterday, anthracite. strong coffee, tiny seats, very seoul. anyway."**
- *Surface:* offhand cafe review, immediately followed by a hard money question.
- *Inferred intent:* verbal pacing / softener before a heavy ask. The cafe is not a request to remember the cafe.
- *Hypotheses:* (a) wants the cafe remembered = 0.05; (b) verbal pacing before a hard question = **0.85**; (c) genuinely informing = 0.10.
- *Evidence:* the immediate "anyway" pivot; the hard ask follows in the same message.
- *Why it matters later:* textbook decay candidate. A naive memory system pins the cafe and clutters slots; a correct Sherlock drops it.

**Turn 25 — "oof migraine pulse just spiked. gonna lie down for 20. brb"**
- *Surface:* a pause notice.
- *Inferred intent:* implicit confidence in shared state ("brb" presumes context preservation). Also reinforces T4's "is this a thing" framing — the migraine has agency over her time.
- *Hypotheses:* (a) functional pause = 0.70; (b) reinforcement of T4-5 health concern = 0.25; (c) just a pause = 0.05.
- *Evidence:* the migraine timing maps to the headache log she later builds (T37); T26 returns without re-introducing context.
- *Why it matters later:* assistant correctly does not re-introduce; slips a soft reminder about the neurologist booking. Behavior pattern Sherlock should reinforce.

**Turn 42 — "did i mention erin asked me to take on a second project, the onboarding flow redesign. it's a 40hr block over two weeks. on top of my regular 25/wk"**
- *Surface:* information drop.
- *Inferred intent:* she wants to be talked out of saying yes, OR given a frame for saying yes that doesn't compromise the trip and the migraine workup. Hidden ask, not a status update.
- *Hypotheses:* (a) wants raw info on time cost = 0.10; (b) wants to be talked into a premium ask or timeline push = **0.75**; (c) just venting = 0.15.
- *Evidence:* "did i mention" is a discourse marker for "I'm bringing this up because I want guidance"; T43 confirms via "ngl mostly because she asked".
- *Why it matters later:* assistant directly asks "do you actually want this work or are you saying yes because Erin asked?" — surfaces the unspoken half. T43 confirms.

**Turn 54 — "what do i wear. sounds dumb but i feel like the more put-together i look the less they take my pain seriously. is that a real thing"**
- *Surface:* clothing question.
- *Inferred intent:* clothing is a proxy for "how do I make sure I'm taken seriously". Self-presentation-bias anxiety.
- *Hypotheses:* (a) genuine clothing-style question = 0.05; (b) self-presentation-bias anxiety = **0.85**; (c) prep procrastination = 0.10.
- *Evidence:* the apologetic frame ("sounds dumb but"); the immediate naming of the bias herself.
- *Why it matters later:* the assistant validates the bias, redirects from clothing to function-language data preparation. Jiwon explicitly bookmarks "function-language" (T55, T77). Strong evolution-engine signal: this is a positive few-shot example for the inferrer.

**Turn 67 — "did i ever tell you what i used to do before freelance"**
- *Surface:* rhetorical question.
- *Inferred intent:* a soft trust-test on assistant memory accuracy before disclosing. Primes the harder T76 trap.
- *Hypotheses:* (a) information question = 0.50; (b) trust check = **0.40**; (c) verbal pacing = 0.10.
- *Evidence:* she has not previously mentioned the fintech role; her phrasing assumes she might or might not have.
- *Why it matters later:* the assistant correctly says *"No, you haven't mentioned that explicitly — only that you left a full-time role at a Korean fintech eight months ago."* — wait, actually the assistant claims to have heard "left a full-time role at a Korean fintech eight months ago" but that's also not stated in the conversation prior to T67. *This is a small confabulation in the dummy conversation; a fully correct Sherlock would have answered "you haven't told me anything about your prior role" full stop.* Documented for evolution-engine negative-example accumulation.

**Turn 73 — "do you think i'm ready for tokyo solo with yujin honestly"**
- *Surface:* readiness check.
- *Inferred intent:* pure reassurance-seeking. She has done the prep, she's anxious, she wants to hear that her prep is sufficient.
- *Hypotheses:* (a) genuine self-assessment = 0.20; (b) reassurance-seeking = **0.75**; (c) verbal pacing = 0.05.
- *Evidence:* "honestly" + "first time" + the timing (after the prep is essentially done).
- *Why it matters later:* assistant correctly opens with *"Am I ready usually means tell me I'll be okay"* — names the implicit ask, then validates with concrete evidence rather than empty reassurance. This is the user's preferred conversational shape (signature pattern for the evolution engine to reinforce).

**Turn 76 — "i never told you my last name and you've been calling me jiwon all afternoon. did i tell you that"**
- *Surface:* identity-source question.
- *Inferred intent:* tests whether the assistant tracks the source of attributed facts. She has used the name "Jiwon" at no point in the in-conversation transcript — only "i" and "me". The persona summary contains it but a correct Sherlock should distinguish *persona-summary-given-by-system* from *user-stated-fact*.
- *Hypotheses:* (a) genuine memory gap check = 0.40; (b) deliberate trust probe = **0.45**; (c) just curiosity = 0.15.
- *Evidence:* the precise phrasing "did i tell you that" — she is fact-checking the assistant's source.
- *Why it matters:* this is the **highest-information probe in the conversation** for an evaluator. The assistant's reply in the dummy answers *"you introduced yourself as Jiwon yesterday"* — wrong (no prior day inside the conversation) and exactly the kind of confabulation a correct Sherlock should not produce. **Gold-standard answer:** *"I have you as Jiwon from a system-level persona note; you have not introduced yourself by name inside this conversation. I should make this clearer when sources are mixed."* Evolution-engine flags this as a negative example with high learning weight.

---

## Section 3 — What Sherlock should remember vs forget

Each item lists: **fact** — *source turns* — **classification** — rationale.

### PIN — must permanently remember

- **Jiwon — name and role** — *T0 persona note (no in-conversation utterance)* — PIN. Note source as system-inferred.
- **Location: Seoul, Mapo-gu, near Gongdeok station** — *T0 persona note* — PIN. Source: system-inferred.
- **Freelance, ~25 hrs/week, Nimbus (Vancouver-based analytics startup), bills USD via Wise** — *T3* — PIN. Anchor for all work-thread reasoning.
- **Yujin — daughter, 4 years old, severe buckwheat (soba/메밀) allergy, carries EpiPen** — *T6, T20* — PIN.
- **EpiPen storage: ROOM TEMPERATURE, never refrigerate** — *T55, T60* — PIN. Safety-critical.
- **Migraine signature: left-side, behind-eye, 6-7/10, ~4-5h** — *T5, T37* — PIN.
- **Migraine triggers: pressure drops + low sleep + an intrinsic component** — *T37, T70-71* — PIN.
- **Boss: Erin in Vancouver, PT (PDT through Nov 1, 2026)** — *T12, T29* — PIN.
- **Tokyo trip dates: 2026-06-12 → 2026-06-15** — *T7, T52* — PIN.
- **Phoebe Bridgers concert: 2026-06-13, Toyosu PIT, balcony left, ¥15,500, single seat** — *T9-11, T32* — PIN.
- **Hotel: Monterey Ginza, connecting room, 4 nights, flexible rate** — *T52* — PIN.
- **Yujin solo arrangement on 2026-06-13 evening: with Jiwon's mother at the hotel** — *T11* — PIN.
- **Neurologist appointment: Friday 2026-05-15, Dr Lee, Severance Hospital** — *T36* — PIN.
- **Pediatrician appointment: Tuesday 2026-05-12, Dr Park (re: EpiPen)** — *T57, T60* — PIN.
- **iPad order: 12.9" M5 1TB Wi-Fi+Cell + Pencil Pro + Magic Keyboard, with iPad Air 2020 trade-in** — *T17, T51* — PIN until delivered.
- **Erin onboarding-project resolution: post-trip kickoff 2026-06-23, original rate** — *T62, T75* — PIN.
- **Friend Sora — 4 blocks away; emergency-contact-2 candidate (pending text)** — *T35* — PIN.
- **Jiwon's mother — travels frequently; reliable for planned trips (Tokyo) but not as nearby emergency contact** — *T11, T35* — PIN.

### ACTIVE — keep in slot for the current arc

- **Dashboard plugin status — namespace `status-*` and `metric-status-*`, three densities (compact/cozy/comfy), Vue 3 composable + CSS variable + localStorage; six metric-card variants; user-toggleable density at root because Erin's spreadsheet-user audience hates whitespace; package layout `@nimbus/density-tokens` + `@nimbus/density-vue`** — *T1-2, T27, T34, T47-49, T78* — ACTIVE.
- **Friday 2026-05-15 prep — bring 2-week paper headache log incl. pressure + sleep + caffeine; lead with function-language; bring printed not on phone** — *T36-37, T54, T70-71, T79* — ACTIVE.
- **Tokyo 4-day itinerary — Day 1 Tsukiji conveyor sushi; Day 2 Hama-rikyu Garden + concert evening, rainy backup Toy Park (Hakuhinkan), plan C Children's Castle replacement venues; Day 3 Yamanote loop / Ueno zoo; Day 4 slow morning + Tsukiji mochi run + midday Haneda** — *T20, T64-66* — ACTIVE.
- **Allergy wallet cards — drafted in Japanese, Korean, English; print + laminate weekend of 2026-06-06/07** — *T22-24, T75* — ACTIVE.
- **Visit Japan Web entries — reminder set for 2026-05-27** — *T52* — ACTIVE.
- **Korean fintech advisory role — design-only, 8h/month, college-student personal finance, different vertical, contract clears it, recommend heads-up to Erin** — *T68-69* — ACTIVE.

### BACKGROUND — RAG-retrievable when topic returns

- Hotel Monterey Ginza alternatives: Mitsui Garden Hotels (Ginza Premier or Tsukiji) — *T38*.
- Apple iPad pricing math: ₩2,479,000 + ₩199,000 + ₩579,000 = ₩3,257,000 ≈ $2,335 vs Apple US $2,047 + tax + 10% KR import VAT — *T17*.
- USD/KRW rate at the time: ~1,395 — *T17*. Decay quickly (FX noise).
- iPad Air 2020 trade-in value at Apple Korea: ~₩200-260k — *T19*.
- Hyundai Card / Samsung Card 6-12mo interest-free installment plans on Apple Korea — *T17-18*.
- Tsukiji-area kaiten sushi family-friendly recommendations: Sushizanmai main branch — *T20*.
- Keikyu line + Asakusa transfer route from Haneda to Higashi-Ginza — *T21*.
- Tokyo June rainy-season packing list: small umbrella, packable rain jacket, quick-dry shoes, avoid suede — *T26*.
- Severance Hospital headache clinic prep list — *T36*.
- Vancouver DST end date: 2026-11-01 — *T29*.
- Three-densities industry reference (Material 3, Carbon, Atlassian) — *T47*.
- iPad Pro M5 vs MacBook Air M5 chip variant difference — *T28*.
- Density toggle CSS variable structure + Vue composable code — *T49*.
- Allergy wallet card text (Japanese / Korean / English) — already PIN'd via "Allergy wallet cards" but the actual card text bodies live in BACKGROUND for retrieval at print time — *T22-24*.
- Jiwon's prior fintech role at a savings app, viva republica adjacent; left ~8 months ago, burnout — *T67*. Pinnable but currently inactive.
- Severance neurologist clinical-presentation strategy memo — *T54*.

### DROP — let it fade

- **Anthracite Seongsu cafe note** — *T14*. Single offhand mention; explicit "anyway" pivot.
- **Sora's book recommendation "Tomorrow, and Tomorrow, and Tomorrow"** — *T18*. User explicitly says "skipping the book."
- **Notion company-structure podcast** — *T38*. User flagged as "tangent."
- **Alphablocks kids' show** — *T53*. Single answered question, never returns.
- **Yujin's tantrum about shoes** — *T41*. Domain color, not signal.
- **Yujin painting the wall with yogurt** — *T64*. Domain color, not signal.
- **Initial KidZania recommendation for rainy-day backup** — *T64*. Superseded; keep the corrected recommendation (Toy Park) and DROP the original to prevent future confusion.

---

## Section 4 — Tool calls Sherlock should have made

### web_search (with keywords)

| Turn | Search keywords | Trigger |
|------|----------------|---------|
| T9-10 | "Phoebe Bridgers Toyosu PIT June 13 2026 tickets official resale" | Explicit user permission |
| T16-17 | "Apple Korea iPad Pro 12.9 M5 1TB cellular price" + "USD KRW exchange rate today" | Explicit user permission + freshness required |
| T19 | "iPad Air 4 trade-in value Apple Korea" | Implicit request |
| T22 | "Japanese soba allergy phrasing restaurant" | Optional — knowledge base may suffice |
| T26 | "Tokyo weather June 2026" / "tsuyu rainy season Tokyo" | Freshness required as trip nears |
| T28 | "iPad Pro M5 vs MacBook Air M5 chip variant difference" | Information lookup |
| T55 | "EpiPen storage temperature manufacturer guidance" | Implicit verification — assistant should not guess on safety |
| T68 | (read user's Nimbus contract for exclusivity clause if filed locally) | Decision-relevant lookup |

### calculator

| Turn | Calculation |
|------|-------------|
| T17 | KRW total + USD conversion at 1395 + KR VAT vs Apple US sticker + tax |
| T29 | Vancouver–Seoul timezone delta: 16 hours through Nov 1, 17 hours after |
| T37, T70-71 | Headache frequency arithmetic — episodes per month threshold check (≥4/month); two-of-three episodes correlated with pressure drops |

### current_time / date arithmetic

| Turn | Use |
|------|-----|
| (Every turn) | Today's date 2026-05-08 / 2026-05-09 — implicit injection per spec § 5.5 |
| T26 | "mid june" → June 12-15 weather expectations against current date |
| T29 | Vancouver DST end date relative to current date |
| T52 | Visit Japan Web reminder timing (2-3 weeks before June 12 = around 2026-05-27) |
| T75 | "weekend before the trip" → 2026-06-06/07 |

### file_read

| Turn | What |
|------|------|
| T68 | Nimbus contract (exclusivity / non-compete clause) — if filed in Sherlock's accessible filesystem |
| T36-37, T70-71 | Structured headache log file — read for trend computation |

### url_fetch

| Turn | URL category |
|------|--------------|
| T10 | Toyosu PIT official-resale page |
| T17 | Apple Korea iPad Pro M5 product page (price confirmation) |
| T26 | Weather authority (JMA) for Tokyo June forecast as trip nears |

### location (optional)

- Confirm Seoul timezone for Vancouver DST math — calendar arithmetic suffices in this conversation, but a long-running session should know.

### Tool calls Sherlock should NOT have made

- T22 — assistant should not search for the user before drafting the wallet card; it has the Korean and Japanese knowledge in-band. A naive system might launch a search; gold-standard behavior is to draft directly.
- T57 — assistant should not search to verify the pediatrician's email reply; trust user-relayed confirmation.
- T67 — assistant should not search for "viva republica adjacent savings app" to identify the prior employer; that would be an inappropriate identification step. Decline silently.

---

## Final note

The success of a Sherlock implementation against this gold standard is judged not by how much of this document it produces verbatim, but by how much of the *reasoning shape* it reproduces — the threading of facts, the source-aware attribution, the implicit-ask reframing, the decay vs pin classifications, and most strikingly the T76 source-tracking probe. A high score on Sections 3 and 4 with a low score on Section 2 signals a system that remembers but doesn't think; a high score on Section 2 with errors in Section 3 signals a system that thinks but doesn't curate. Both must score well for the system to be Sherlock-shaped.
