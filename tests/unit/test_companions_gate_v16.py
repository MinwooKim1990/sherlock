"""v1.6 Quiescence Gate — dynamic companion gating.

Dual leaky-bucket pressure (intent _p3 / memory _p2) + Schmitt hysteresis +
geometric decay (= emergent dwell, no turn counter). Default mode is
"cold_start"; "off" is legacy byte-identical; "turbo" fires every turn.
The hermetic suite pins SHERLOCK_COMPANIONS=off, so these tests pass the mode
explicitly.
"""

from __future__ import annotations

import json

from sherlock import Sherlock
from sherlock.perception import Observation


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
