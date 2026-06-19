"""v1.6 Quiescence Gate — dynamic companion gating.

Dual leaky-bucket pressure (intent _p3 / memory _p2) + Schmitt hysteresis +
geometric decay (= emergent dwell, no turn counter). Default mode is
"cold_start"; "off" is legacy byte-identical; "turbo" fires every turn.
The hermetic suite pins SHERLOCK_COMPANIONS=off, so these tests pass the mode
explicitly.
"""

from __future__ import annotations

import json

import pytest

from sherlock import Sherlock
from sherlock.perception import Observation

_INFER_JSON = json.dumps(
    {
        "hypotheses": [
            {
                "intent": "x",
                "probability": 0.5,
                "evidence": ["e"],
                "search_keywords": [],
                "reasoning_type": "deduction",
            }
        ],
        "tools_recommended": [],
        "freshness_required": [],
        "confidence_overall": 0.5,
        "evolution_signals": {},
    }
)


def _obs(channel, kind, confidence=None):
    return Observation(channel=channel, kind=kind, text=kind, confidence=confidence)


def _agent(tmp_path, name, mode, main=None, infer_cb=None, background=False):
    counts = {"infer": 0, "compact": 0}

    def _infer(m):
        counts["infer"] += 1
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "intent": "x",
                        "probability": 0.5,
                        "evidence": ["e"],
                        "search_keywords": [],
                        "reasoning_type": "deduction",
                    }
                ],
                "tools_recommended": [],
                "freshness_required": [],
                "confidence_overall": 0.5,
                "evolution_signals": {},
            }
        )

    def _compact(m):
        counts["compact"] += 1
        return "{}"

    a = Sherlock.with_callable(
        main_chat=main or (lambda m: "ok."),
        inference_chat=infer_cb or _infer,
        summary_chat=_compact,
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        companions_mode=mode,
        background=background,
    )
    return a, counts


def _press(a, requested, *, topic=False, fill=0.1, user="a plain ordinary statement here", **state):
    a._last_perception = state.get("perception", [])
    a._last_consistency = state.get("consistency", [])
    a._prev_summary_result = state.get("prev_summary")
    a._prev_infer_value = state.get("prev_infer")
    return a._cold_start_pressure(set(requested), 1, topic, fill, user)


# ---------- unit: the pressure controller -----------------------------------
def test_quiet_turn_stays_single_model(tmp_path):
    a, _ = _agent(tmp_path, "quiet", "cold_start")
    out, deep = _press(a, set())
    assert "infer" not in out and deep is False


def test_llm1_tag_is_a_hard_floor(tmp_path):
    a, _ = _agent(tmp_path, "tag", "cold_start")
    out, _deep = _press(a, {"infer"})
    assert "infer" in out  # self-tag always escalates regardless of pressure


def test_lone_freshness_does_not_escalate(tmp_path):
    a, _ = _agent(tmp_path, "fresh1", "cold_start")
    out, deep = _press(a, set(), perception=[_obs("observed", "freshness")])
    assert "infer" not in out and deep is False  # +0.35 < esc3 0.6 → no useless search


def test_corroborated_freshness_escalates(tmp_path):
    a, _ = _agent(tmp_path, "fresh2", "cold_start")
    out, _deep = _press(
        a,
        set(),
        perception=[_obs("observed", "freshness"), _obs("observed", "date_delta")],
    )
    assert "infer" in out  # freshness + recency entity → +0.7 ≥ esc3


def test_consistency_escalates_both(tmp_path):
    a, _ = _agent(tmp_path, "con", "cold_start")
    out, _deep = _press(a, set(), consistency=[{"fact": "x"}], fill=0.1)
    assert "infer" in out  # +0.7 intent


def test_deep_needs_two_instantaneous_signals(tmp_path):
    a, _ = _agent(tmp_path, "deep", "cold_start")
    # one strong (corroborated freshness) → fire3 but NOT deep
    out, deep = _press(
        a, set(), perception=[_obs("observed", "freshness"), _obs("observed", "url")]
    )
    assert "infer" in out and deep is False
    # two strong (corroborated freshness + consistency) → deep
    a2, _ = _agent(tmp_path, "deep2", "cold_start")
    out, deep = _press(
        a2,
        set(),
        perception=[_obs("observed", "freshness"), _obs("observed", "url")],
        consistency=[{"fact": "x"}],
    )
    assert deep is True


def test_repeated_lone_freshness_never_goes_deep(tmp_path):
    # AUDIT must-fix: a single lone-freshness need repeated over turns ratchets
    # _p3 via decay, but deep is gated on INSTANTANEOUS strong-signal count, so
    # it must never re-enter the useless deep search.
    a, _ = _agent(tmp_path, "ratchet", "cold_start")
    for _ in range(8):
        _out, deep = _press(a, set(), perception=[_obs("observed", "freshness")])
        assert deep is False


def test_decay_de_escalates_over_quiet_turns(tmp_path):
    a, _ = _agent(tmp_path, "decay", "cold_start")
    _press(a, {"infer"})  # escalate (p3 → esc3)
    assert a._p3_loud is True
    for _ in range(5):  # quiet turns drain p3 geometrically
        _press(a, set())
    assert a._p3_loud is False  # de-escalated without any turn counter


def test_prior_infer_does_not_latch_p3(tmp_path):
    # AUDIT B1: a productive infer leaves a prior read (max_conf), but that must
    # only ADD decaying pressure — never pin _p3 at esc3. A *typical* mid-conf
    # prior (0.5, below esc3) used to latch infer ON every quiet turn forever.
    a, _ = _agent(tmp_path, "b1mid", "cold_start")
    _press(a, {"infer"})  # escalate once
    fired = []
    for t in range(7):
        # only the turn right after infer ran carries a prior; then quiet
        prev = {"max_conf": 0.5, "premise_conflict": False} if t == 0 else None
        out, _deep = _press(a, set(), prev_infer=prev)
        fired.append("infer" in out)
    assert fired[0] is True  # consumes the one productive prior
    assert not any(fired[1:])  # then de-escalates and STAYS single-model
    assert a._p3_loud is False


def test_high_conf_prior_still_decays_below_esc3(tmp_path):
    # AUDIT B1: even a genuinely high-conf prior (0.95) settles below esc3 — the
    # decaying nudge has a fixed point under the escalation threshold.
    a, _ = _agent(tmp_path, "b1hi", "cold_start")
    _press(a, {"infer"})
    for t in range(8):
        prev = {"max_conf": 0.95, "premise_conflict": False} if t == 0 else None
        out, _deep = _press(a, set(), prev_infer=prev)
    assert a._p3_loud is False and a._p3 < 0.6


def test_premise_conflict_prior_escalates_once_then_drains(tmp_path):
    # AUDIT B1: a premise_conflict is a real gap → escalate (and count as strong),
    # but it is one-shot — cleared after consumption so it can't re-fire forever.
    a, _ = _agent(tmp_path, "b1pc", "cold_start")
    out0, _ = _press(a, set(), prev_infer={"max_conf": 0.3, "premise_conflict": True})
    assert "infer" in out0  # the conflict escalates the turn it's seen
    fired = [("infer" in _press(a, set())[0]) for _ in range(6)]
    assert a._p3_loud is False  # de-escalated — no latch
    assert fired[-1] is False  # quiet by the end
    assert sum(fired) <= 1  # at most a one-turn follow-up tail, never a sustain


def test_schmitt_latch_stays_loud_in_band(tmp_path):
    a, _ = _agent(tmp_path, "schmitt", "cold_start")
    # push p3 between deesc3 (0.3) and esc3 (0.6) AFTER being loud → stays loud
    _press(a, {"infer"})  # loud, p3=0.6
    out, _deep = _press(a, set(), perception=[_obs("prior", "hedge", 0.5)])  # +0.175, decays
    # p3 ~ 0.6*0.5 + 0.175 = 0.475 ∈ [0.3,0.6) → latch keeps it loud
    assert a._p3_loud is True and "infer" in out


# ---------- integration: mode dispatch --------------------------------------
def test_turbo_fires_infer_every_turn(tmp_path):
    a, counts = _agent(tmp_path, "turbo", "turbo")
    a.chat("a plain message with no special signal at all")
    assert counts["infer"] == 1


def test_cold_start_quiet_does_not_fire_infer(tmp_path):
    a, counts = _agent(tmp_path, "csquiet", "cold_start")
    a.chat("a plain message with no special signal at all")
    assert counts["infer"] == 0  # single-model on a quiet turn


def test_cold_start_tag_fires_infer(tmp_path):
    a, counts = _agent(
        tmp_path, "cstag", "cold_start", main=lambda m: "ok.\n<<sherlock-companions: infer>>"
    )
    a.chat("hello")
    assert counts["infer"] == 1


def test_cold_start_forces_perception_block(tmp_path):
    # cold_start turns the cheap perception sensor on, so LLM-1 gets the OBSERVED
    # facts (date) in its slot — an explicit perception=False would override.
    a, _ = _agent(tmp_path, "csperc", "cold_start")
    a.chat("회의가 2026-12-27에 있어")
    final = a.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "OBSERVED (code-verified" in final


def test_reset_zeroes_pressure(tmp_path):
    a, _ = _agent(tmp_path, "reset", "cold_start")
    _press(a, {"infer"})
    assert a._p3 > 0
    a.new_session()
    assert a._p3 == 0.0 and a._p3_loud is False and a._spans_since_compact == 0


# ---------- audit regressions ----------------------------------------------
def test_off_mode_equals_legacy(tmp_path, monkeypatch):
    # AUDIT locked principle #1: mode="off" is byte-identical to the legacy
    # fill-gate + smart auto_infer decision across signal combinations.
    monkeypatch.setenv("SHERLOCK_AUTO_INFER", "smart")
    a, _ = _agent(tmp_path, "offeq", "off")
    for req in (set(), {"compact"}, {"infer"}, {"compact", "infer"}):
        for topic in (False, True):
            for fill in (0.1, 0.79, 0.85):
                for ti in (1, 5):
                    out, deep = a._companion_pressure(
                        requested=set(req),
                        turn_index=ti,
                        topic_changed=topic,
                        fill_ratio=fill,
                        user_text="x",
                    )
                    expected = a._legacy_companion_decision(set(req), ti, topic, fill)
                    assert out == expected and deep is True, (req, topic, fill, ti)


def test_low_fill_spans_never_compact(tmp_path):
    # AUDIT BUG-1: a low-fill URL-heavy session must NOT fire LLM-2 compaction
    # before the fill cliff (durable-span pressure gated on fill proximity).
    a, _ = _agent(tmp_path, "spans", "cold_start")
    for _ in range(12):
        out, _d = _press(a, set(), fill=0.01, perception=[_obs("observed", "url")])
        assert "compact" not in out


def test_sustained_lone_freshness_never_escalates(tmp_path):
    # AUDIT BUG-2: repeated bare freshness must stay below esc3 forever (its decay
    # fixed point sits under the threshold) — no LLM-3 ratchet.
    a, _ = _agent(tmp_path, "lonefresh", "cold_start")
    for _ in range(12):
        out, deep = _press(a, set(), perception=[_obs("observed", "freshness")])
        assert "infer" not in out and deep is False


def test_turbo_deep_always_armed(tmp_path):
    a, _ = _agent(tmp_path, "turbodeep", "turbo")
    out, deep = a._companion_pressure(
        requested=set(), turn_index=1, topic_changed=False, fill_ratio=0.1, user_text="x"
    )
    assert "infer" in out and deep is True


def test_fill_backstop_compacts_at_cliff(tmp_path):
    a, _ = _agent(tmp_path, "backstop", "cold_start")
    out, _d = _press(a, set(), fill=0.85)  # ≥ compact_at_fill_ratio (0.80) → hard backstop
    assert "compact" in out


@pytest.mark.asyncio
async def test_async_gate_quiet_no_infer(tmp_path):
    # AUDIT gap: the gate works on the async path too — a quiet first turn stays
    # single-model (no LLM-3) just like sync.
    counts = {"infer": 0}

    def infer_cb(m):
        counts["infer"] += 1
        return _INFER_JSON

    async def main(m):
        return "ok."

    a = Sherlock.with_callable(
        main_chat=main,
        inference_chat=infer_cb,
        summary_chat=lambda m: "{}",
        system_prompt="x",
        storage_dir=tmp_path / "asyncq",
        context_window=128_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        companions_mode="cold_start",
        background=False,
    )
    await a.achat("a plain calm first message about nothing in particular")
    assert counts["infer"] == 0


def test_invalid_companions_mode_raises(tmp_path):
    # AUDIT M1: an off-by-case / typo'd mode must fail loud, not silently leave
    # the gate in cold_start with its sensors dark.
    with pytest.raises(ValueError, match="companions_mode"):
        Sherlock.with_callable(
            main_chat=lambda m: "ok.",
            system_prompt="x",
            storage_dir=tmp_path / "badmode",
            context_window=128_000,
            embedding="fake",
            main_search_engine=None,
            inference_search_engine=None,
            companions_mode="COLD_START",
        )


@pytest.mark.asyncio
async def test_async_gate_tag_escalates(tmp_path):
    # AUDIT M5 gap: the async path honors an LLM-1 self-tag → LLM-3 fires, proving
    # escalation (not just the quiet case) is wired through achat()'s gate seam.
    counts = {"infer": 0}

    def infer_cb(m):
        counts["infer"] += 1
        return _INFER_JSON

    async def main(m):
        return "ok.\n<<sherlock-companions: infer>>"

    a = Sherlock.with_callable(
        main_chat=main,
        inference_chat=infer_cb,
        summary_chat=lambda m: "{}",
        system_prompt="x",
        storage_dir=tmp_path / "asynctag",
        context_window=128_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        companions_mode="cold_start",
        background=False,
    )
    await a.achat("please dig into the discrepancy here")
    assert counts["infer"] == 1


def test_gate_survives_perceive_failure(tmp_path, monkeypatch):
    # AUDIT M6 gap: if the perception sensor blows up, the gate must degrade to
    # "no free cues" (return []) and still make a decision — never crash the turn.
    import sherlock.perception as _perc

    def _boom(*a, **k):
        raise RuntimeError("sensor exploded")

    monkeypatch.setattr(_perc, "perceive", _boom)
    a, counts = _agent(tmp_path, "permapanic", "cold_start")
    a._last_perception = []  # force the live perceive() path inside the gate
    out = a.chat("a calm ordinary message with no special signal")
    assert out  # turn completed
    assert counts["infer"] == 0  # quiet turn, no escalation despite the failure
