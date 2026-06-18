"""v1.5 Stage 1 — stdlib perception layer.

Two halves: (1) the deterministic primitives (date/script/arithmetic/spans/
code/freshness/discourse) computed correctly, and (2) the slot wiring —
OFF (default) is byte-inert, ON injects the OBSERVED/PRIOR block.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from sherlock.config import Config, PerceptionConfig
from sherlock.perception import Observation, perceive, render_observations

NOW = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)


def _kinds(obs):
    return {o.kind for o in obs}


def _by_kind(obs, kind):
    return [o for o in obs if o.kind == kind]


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------
def test_iso_date_delta_weekday():
    obs = perceive("let's meet on 2026-12-27 in Tokyo", now=NOW)
    dd = _by_kind(obs, "date_delta")
    assert dd, "expected a date_delta observation"
    target = date(2026, 12, 27)
    delta = (target - NOW.date()).days
    assert "2026-12-27" in dd[0].text
    assert target.strftime("%A") in dd[0].text  # correct weekday
    assert f"{delta} days from today" in dd[0].text
    assert dd[0].channel == "observed"


def test_korean_date_year_assumed():
    obs = perceive("12월 27일에 도쿄에서 뭐하지?", now=NOW)
    dd = _by_kind(obs, "date_delta")
    assert dd
    assert "2026-12-27" in dd[0].text
    assert "year assumed" in dd[0].text  # no year given → flagged, not silent


def test_past_date_phrasing():
    obs = perceive("the incident on 2026-01-05", now=NOW)
    dd = _by_kind(obs, "date_delta")
    assert dd and "ago" in dd[0].text


def test_business_days_exact():
    # Lock the business-day count against a brute-force weekday walk.
    from sherlock.perception.core import _business_days

    def brute(d0, d1):
        from datetime import timedelta

        n, d = 0, d0
        while d < d1:
            d += timedelta(days=1)
            if d.weekday() < 5:
                n += 1
        return n

    for target in (date(2026, 12, 27), date(2026, 7, 1), date(2027, 3, 15)):
        assert _business_days(NOW.date(), target) == brute(NOW.date(), target), target


def test_leap_day_invalid_date_dropped():
    # An explicit invalid date (Feb 29 of a non-leap year) must be dropped
    # silently, never crash or render a bogus fact.
    assert not _by_kind(perceive("meeting 2025-02-29", now=NOW), "date_delta")  # 2025 not leap
    assert _by_kind(perceive("meeting 2024-02-29", now=NOW), "date_delta")  # 2024 is leap


def test_may_is_not_a_month_when_lowercase_and_no_year():
    # "may" as the ordinary verb must NOT become a May date.
    obs = perceive("i may 5 times this week", now=NOW)
    assert not _by_kind(obs, "date_delta")
    # But capitalized "May 5", or lowercase "may" with an explicit year+day, is real.
    assert _by_kind(perceive("see you May 5", now=NOW), "date_delta")
    assert _by_kind(perceive("deadline may 5 2027", now=NOW), "date_delta")


# --------------------------------------------------------------------------
# scripts / locale
# --------------------------------------------------------------------------
def test_script_korean():
    obs = perceive("안녕하세요 도와주세요", now=NOW)
    s = _by_kind(obs, "script")
    assert s and "Korean" in s[0].text


def test_script_japanese():
    obs = perceive("クリスマスのイベント", now=NOW)
    s = _by_kind(obs, "script")
    assert s and "Japanese" in s[0].text


def test_script_silent_for_plain_english():
    obs = perceive("hello there, how are you?", now=NOW)
    assert not _by_kind(obs, "script")


# --------------------------------------------------------------------------
# arithmetic (exact Decimal)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "msg,expected",
    [
        ("what is 1234 * 5.5", "1234 * 5.5 = 6787"),
        ("0.1 + 0.2 please", "0.1 + 0.2 = 0.3"),  # beats float 0.30000000004
        ("100 - 37 = ?", "100 - 37 = 63"),
        ("(3 + 4) * 2", "(3 + 4) * 2 = 14"),
    ],
)
def test_arithmetic_exact(msg, expected):
    obs = perceive(msg, now=NOW)
    a = _by_kind(obs, "arithmetic")
    assert a, f"no arithmetic obs for {msg!r}"
    assert expected in a[0].text


def test_arithmetic_skips_dates_ips_versions():
    for msg in ("2026-12-27", "ping 192.168.0.1", "version 1.2.3", "3-day trip"):
        assert not _by_kind(perceive(msg, now=NOW), "arithmetic"), msg


def test_arithmetic_no_scinotation_or_separator_truncation():
    # AUDIT A1: a span sliced out of sci-notation / digit-grouped tokens must
    # NOT be stated as a fact ("1e10 * 2" must not become "10 * 2 = 20").
    for msg in ("scale it by 1e10 * 2", "1.5e3 / 3", "3.14e2 - 14", "1_000 + 1"):
        assert not _by_kind(perceive(msg, now=NOW), "arithmetic"), msg


def test_arithmetic_no_time_ranges():
    # AUDIT A2: "9:00 - 17:00" must not subtract the minute fragments.
    for msg in ("my shift is 9:00 - 17:00", "from 10:30 - 12:45", "13:00 + 30 min"):
        assert not _by_kind(perceive(msg, now=NOW), "arithmetic"), msg


def test_arithmetic_no_idioms():
    # AUDIT A5: bare N/N idioms/ratios are not arithmetic.
    for msg in ("it's 24/7 support", "I rate it 9/10", "a 50/50 split"):
        assert not _by_kind(perceive(msg, now=NOW), "arithmetic"), msg


def test_arithmetic_no_double_operator_fragment():
    # AUDIT A6: "--5 + 1" is a truncation artifact, not a clean fact.
    assert not _by_kind(perceive("weird --5 + 1 thing", now=NOW), "arithmetic")
    # but a single unary minus is legitimate
    a = _by_kind(perceive("compute -5 + 1", now=NOW), "arithmetic")
    assert a and "-4" in a[0].text


def test_date_skips_version_strings():
    # AUDIT A3: dotted/slashed version strings must NOT become calendar dates.
    for msg in (
        "release 2024.2.29 notes",
        "build 2025.06.30.1234",
        "see 2026.12.27.00",
        "ISO 9001.2.3 cert",
        "path 2026/12/27/extra",
        "weird 2026-12/27 mix",
    ):
        assert not _by_kind(perceive(msg, now=NOW), "date_delta"), msg


def test_korean_dot_date_still_works():
    # A plausible-year dot date with no version context IS a date (KR format).
    dd = _by_kind(perceive("미팅은 2026.12.27 입니다", now=NOW), "date_delta")
    assert dd and "2026-12-27" in dd[0].text
    # and the CJK-glued case the layer was built for still parses
    assert _by_kind(perceive("회의가 2026-12-27에 있어", now=NOW), "date_delta")


def test_url_email_trim_cjk_particle():
    # AUDIT A4: a glued CJK particle must not be swallowed into the "verified"
    # host / domain.
    urls = _by_kind(perceive("https://example.com에서 확인해", now=NOW), "url")
    assert urls and "example.com" in urls[0].text and "에서" not in urls[0].text
    emails = _by_kind(perceive("user@test.com으로 보내", now=NOW), "email")
    assert emails and "test.com" in emails[0].text and "으로" not in emails[0].text


def test_nil_uuid_no_vnone():
    obs = _by_kind(perceive("id 00000000-0000-0000-0000-000000000000", now=NOW), "uuid")
    assert obs and "vNone" not in obs[0].text


# --------------------------------------------------------------------------
# structural spans + SSRF flag
# --------------------------------------------------------------------------
def test_url_public_and_private():
    obs = perceive("check https://example.com/path and http://localhost:8000/x", now=NOW)
    urls = _by_kind(obs, "url")
    texts = " ".join(u.text for u in urls)
    assert "example.com" in texts
    assert "localhost" in texts and "non-public" in texts


def test_private_ip_flagged():
    obs = perceive("the server is at 10.0.0.5 internally", now=NOW)
    ip = _by_kind(obs, "ip")
    assert ip and "private/internal" in ip[0].text


def test_uuid_detected():
    obs = perceive("trace id 550e8400-e29b-41d4-a716-446655440000", now=NOW)
    assert _by_kind(obs, "uuid")


# --------------------------------------------------------------------------
# code signals
# --------------------------------------------------------------------------
def test_code_fence():
    obs = perceive("look at this:\n```python\nprint(1)\n```", now=NOW)
    cb = _by_kind(obs, "code_block")
    assert cb and "python" in cb[0].text


def test_traceback():
    msg = 'Traceback (most recent call last):\n  File "x.py", line 3\nValueError'
    obs = perceive(msg, now=NOW)
    assert _by_kind(obs, "traceback")


# --------------------------------------------------------------------------
# freshness (strong keywords only)
# --------------------------------------------------------------------------
def test_freshness_stock_price():
    obs = perceive("show me spaceX stock price", now=NOW)
    f = _by_kind(obs, "freshness")
    assert f and "stock price" in f[0].text


def test_freshness_silent_without_strong_keyword():
    obs = perceive("tell me about the history of spacex", now=NOW)
    assert not _by_kind(obs, "freshness")


# --------------------------------------------------------------------------
# discourse
# --------------------------------------------------------------------------
def test_anaphora_korean():
    obs = perceive("그거 어떻게 됐어?", now=NOW)
    assert _by_kind(obs, "anaphora")


def test_hedge():
    obs = perceive("maybe we should refactor this", now=NOW)
    assert _by_kind(obs, "hedge")


def test_anaphora_en_pronominal_only():
    # AUDIT D: fire on pronominal demonstratives, NOT determiners or dummy "it".
    assert _by_kind(perceive("that doesn't work", now=NOW), "anaphora")
    assert _by_kind(perceive("those are nice", now=NOW), "anaphora")
    for msg in ("the meeting is this afternoon", "it works now", "it's 24/7 support"):
        assert not _by_kind(perceive(msg, now=NOW), "anaphora"), msg


def test_freshness_latest_version_is_not_live_data():
    # AUDIT D: "latest version" is a software query, not a live-data request.
    assert not _by_kind(perceive("give me the latest version", now=NOW), "freshness")
    # but "latest news" is genuinely time-sensitive
    assert _by_kind(perceive("latest news please", now=NOW), "freshness")


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def test_render_two_channels():
    obs = [
        Observation("observed", "x", "fact one"),
        Observation("prior", "y", "cue one", confidence=0.6),
    ]
    text = render_observations(obs)
    assert "OBSERVED (code-verified" in text
    assert "PRIOR (probabilistic" in text
    assert "- fact one" in text
    assert "(~0.6) cue one" in text


def test_render_empty():
    assert render_observations([]) == ""


def test_render_cap_prioritizes_observed():
    obs = [Observation("observed", "x", f"f{i}") for i in range(20)]
    obs += [Observation("prior", "y", "cue", confidence=0.5)]
    text = render_observations(obs, max_observations=3)
    assert text.count("- f") == 3
    assert "cue" not in text  # observed filled the cap


# --------------------------------------------------------------------------
# config + slot wiring
# --------------------------------------------------------------------------
def test_config_default_off():
    assert Config.model_construct is not None
    assert PerceptionConfig().enabled is False


def _agent(tmp_path, perception, name):
    from sherlock import Sherlock

    return Sherlock.with_callable(
        main_chat=lambda m: "ok.",
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        main_search_engine=None,
        inference_search_engine=None,
        perception=perception,
    )


def test_slot_off_is_inert(tmp_path):
    agent = _agent(tmp_path, False, "off")
    agent.chat("회의가 2026-12-27에 있어")
    final = agent.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "OBSERVED (code-verified" not in final
    assert agent._last_perception == []


def test_slot_on_injects_observed(tmp_path):
    agent = _agent(tmp_path, True, "on")
    agent.chat("회의가 2026-12-27에 있어")
    final = agent.inspect_last_turn().messages_passed_to_llm1[-1].content
    assert "OBSERVED (code-verified" in final
    assert "2026-12-27" in final
    assert any(o.kind == "date_delta" for o in agent._last_perception)
