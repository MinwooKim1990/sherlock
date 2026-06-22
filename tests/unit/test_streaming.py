"""Phase 1 streaming helper: the playground main reply streams answer tokens
(`on_delta`) and reasoning/"thinking" tokens (`on_reasoning`, from litellm's
normalized `delta.reasoning_content`), stops early on `should_stop()`, and
returns a ModelResponse-shaped object with the full text. Driven against a fake
litellm stream so no live provider/key is needed."""

from __future__ import annotations

import types

import litellm

import playground.providers as prov


def _chunk(content=None, reasoning=None):
    delta = types.SimpleNamespace(content=content, reasoning_content=reasoning)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


def test_stream_emits_content_and_reasoning(monkeypatch):
    def fake_completion(model, messages, stream=False, **kw):
        assert stream is True
        return iter(
            [
                _chunk(reasoning="think "),
                _chunk(reasoning="more"),
                _chunk("Hello"),
                _chunk(" world"),
            ]
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    answers, reasons = [], []
    resp = prov._call_litellm_stream(
        "x/y",
        [{"role": "user", "content": "hi"}],
        answers.append,
        lambda: False,
        on_reasoning=reasons.append,
    )
    assert "".join(answers) == "Hello world"
    assert "".join(reasons) == "think more"
    assert resp.choices[0].message.content == "Hello world"


def test_stream_stops_between_chunks(monkeypatch):
    def fake_completion(model, messages, stream=False, **kw):
        return iter([_chunk("A"), _chunk("B"), _chunk("C")])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    answers = []
    # should_stop flips True after the first chunk → only "A" streams
    flags = {"seen": 0}

    def should_stop():
        flags["seen"] += 1
        return flags["seen"] >= 1

    resp = prov._call_litellm_stream(
        "x/y", [{"role": "user", "content": "hi"}], answers.append, should_stop
    )
    assert answers == ["A"]
    assert resp.choices[0].message.content == "A"


def test_non_reasoning_stream_has_no_reasoning(monkeypatch):
    def fake_completion(model, messages, stream=False, **kw):
        return iter([_chunk("just "), _chunk("text")])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    reasons = []
    prov._call_litellm_stream(
        "x/y",
        [{"role": "user", "content": "hi"}],
        lambda p: None,
        lambda: False,
        on_reasoning=reasons.append,
    )
    assert reasons == []  # a non-reasoning model emits no thinking deltas
