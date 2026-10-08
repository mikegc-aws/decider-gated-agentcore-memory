"""Tests for the gating logic, with a stub decider and a stub memory client.

These deliberately do not touch AWS or either decision endpoint: the point is to
pin the control flow (when do we call the memory service, what reaches the prompt,
what happens when the decider breaks) independently of model behaviour, which is
measured separately by the scripts in scripts/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decider_memory.config import DeciderGateConfig, GateStats  # noqa: E402
from decider_memory.records import render_memory_text  # noqa: E402
from decider_memory.session_manager import (  # noqa: E402
    DeciderGatedAgentCoreMemorySessionManager as Gated,
)


class StubDecider:
    """Returns canned probabilities; records every call for assertions."""

    name = "stub"

    def __init__(self, gate: float = 1.0, relevance: dict[str, float] | None = None, boom=False):
        self.gate_p = gate
        self.relevance = relevance or {}
        self.boom = boom
        self.calls: list[dict] = []

    def ask(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self.boom:
            raise RuntimeError("decider is down")
        if "worth" in questions:
            return {"worth": self.gate_p}
        # Validation batch: match each question back to its memory text.
        out = {}
        for name, q in questions.items():
            instructions = q["instructions"]
            out[name] = next(
                (p for text, p in self.relevance.items() if text in instructions), 0.0
            )
        return out


def make_manager(decider, gate_config, candidates):
    """Build a manager without running the real __init__ (which calls AWS)."""
    manager = Gated.__new__(Gated)
    manager.decider = decider
    manager.gate_config = gate_config
    manager.stats = GateStats()
    manager.config = SimpleNamespace(
        retrieval_config={"/facts/{actorId}": SimpleNamespace(top_k=10, relevance_score=0.2)},
        actor_id="u1",
        session_id="s1",
        memory_id="m1",
        context_tag="user_context",
    )
    def fetch(query):
        # Stand in for the real _fetch_candidates, including the counter it owns,
        # so the stats assertions below mean the same thing they do in production.
        manager.stats.retrieve_calls += 1
        return [dict(c) for c in candidates]

    manager._fetch_candidates = fetch  # type: ignore[method-assign]
    return manager


def make_event(text: str):
    """A minimal stand-in for MessageAddedEvent."""
    messages = [{"role": "user", "content": [{"text": text}]}]
    return SimpleNamespace(agent=SimpleNamespace(messages=messages)), messages


CANDIDATES = [
    {"text": "The user's go-to drink is a decaf oat flat white.", "score": 0.43},
    {"text": "The user writes mostly Python.", "score": 0.36},
]


def test_gate_closed_skips_retrieval_entirely():
    decider = StubDecider(gate=0.05)
    manager = make_manager(decider, DeciderGateConfig(gate_threshold=0.19), CANDIDATES)
    event, messages = make_event("Hello")

    manager.retrieve_customer_context(event)

    assert manager.stats.gate_closed == 1
    assert manager.stats.retrieve_calls == 0, "must not call the memory service"
    assert manager.stats.baseline_retrieve_calls == 1, "but should record what upstream would do"
    assert len(messages[0]["content"]) == 1, "nothing injected"
    assert len(decider.calls) == 1, "gate only, no validation"


def test_gate_open_then_validation_filters_and_injects():
    decider = StubDecider(
        gate=0.62,
        relevance={"decaf oat flat white": 0.50, "mostly Python": 0.20},
    )
    manager = make_manager(decider, DeciderGateConfig(validate_threshold=0.36), CANDIDATES)
    event, messages = make_event("I'd like a drink")

    manager.retrieve_customer_context(event)

    assert manager.stats.retrieve_calls == 1
    assert manager.stats.memories_returned == 2
    assert manager.stats.memories_kept == 1
    assert manager.stats.memories_dropped == 1

    injected = messages[0]["content"][0]["text"]
    assert injected.startswith("<user_context>")
    assert "decaf oat flat white" in injected
    assert "Python" not in injected, "irrelevant memory must not reach the prompt"
    # The user's own text stays last, as upstream guarantees.
    assert messages[0]["content"][-1]["text"] == "I'd like a drink"


def test_validation_batches_into_a_single_call():
    """N memories must cost one round trip, not N."""
    many = [{"text": f"Fact number {i}.", "score": 0.4} for i in range(10)]
    decider = StubDecider(gate=0.9, relevance={f"Fact number {i}.": 0.9 for i in range(10)})
    manager = make_manager(decider, DeciderGateConfig(max_validate=12), many)
    event, _ = make_event("tell me about me")

    manager.retrieve_customer_context(event)

    assert len(decider.calls) == 2, "one gate call + one validation call"
    assert len(decider.calls[1]["questions"]) == 10
    assert manager.stats.memories_kept == 10


def test_validation_splits_when_over_max_validate():
    many = [{"text": f"Fact number {i}.", "score": 0.4} for i in range(7)]
    decider = StubDecider(gate=0.9, relevance={f"Fact number {i}.": 0.9 for i in range(7)})
    manager = make_manager(decider, DeciderGateConfig(max_validate=3), many)
    event, _ = make_event("tell me about me")

    manager.retrieve_customer_context(event)

    # 7 candidates at 3 per call = 3 validation calls, plus the gate.
    assert len(decider.calls) == 4
    assert manager.stats.memories_kept == 7


def test_broken_decider_fails_open_on_the_gate():
    """A dead decider must not cost the agent its memory."""
    decider = StubDecider(boom=True)
    manager = make_manager(decider, DeciderGateConfig(), CANDIDATES)
    event, messages = make_event("I'd like a drink")

    manager.retrieve_customer_context(event)

    assert manager.stats.retrieve_calls == 1, "retrieved anyway"
    assert manager.stats.memories_kept == 2, "kept everything unfiltered"
    assert "decaf oat flat white" in messages[0]["content"][0]["text"]
    # The degradation must be visible: a silent fail-open once made a whole demo
    # run look merely ineffective when every decision had in fact errored.
    assert manager.stats.decider_errors == 2, "gate + validation errors counted"
    assert "decider errors" in manager.stats.report()


def test_decider_none_matches_upstream_behaviour():
    """The control arm: no gate, no validation, everything injected."""
    manager = make_manager(None, DeciderGateConfig(gate=False, validate=False), CANDIDATES)
    event, messages = make_event("Hello")

    manager.retrieve_customer_context(event)

    assert manager.stats.retrieve_calls == 1
    assert manager.stats.memories_kept == 2
    assert manager.stats.memories_dropped == 0
    injected = messages[0]["content"][0]["text"]
    assert "decaf oat flat white" in injected and "Python" in injected


def test_all_memories_rejected_injects_nothing():
    decider = StubDecider(gate=0.9, relevance={})  # everything scores 0.0
    manager = make_manager(decider, DeciderGateConfig(), CANDIDATES)
    event, messages = make_event("what is 2+2")

    manager.retrieve_customer_context(event)

    assert manager.stats.memories_kept == 0
    assert len(messages[0]["content"]) == 1, "no empty context block"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": [{"text": "hi"}]},
        {"role": "user", "content": [{"toolResult": {"content": []}}]},
        {"role": "user", "content": []},
    ],
)
def test_non_user_text_messages_are_ignored(message):
    decider = StubDecider(gate=0.9)
    manager = make_manager(decider, DeciderGateConfig(), CANDIDATES)
    event = SimpleNamespace(agent=SimpleNamespace(messages=[message]))

    manager.retrieve_customer_context(event)

    assert decider.calls == []
    assert manager.stats.turns == 0


# ------------------------------------------------------------------- rendering


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The user is vegetarian.", "The user is vegetarian."),
        (
            '{"context":"Ordering a drink or beverage",'
            '"preference":"Dislikes sparkling water","categories":["x"]}',
            "When ordering a drink or beverage: Dislikes sparkling water",
        ),
        ('{"preference":"Vegetarian diet"}', "Vegetarian diet"),
        # Acronyms must survive the lower-casing of the first character.
        (
            '{"context":"YVR departures","preference":"Aisle seat"}',
            "When YVR departures: Aisle seat",
        ),
        # Unparseable input is passed through rather than mangled.
        ("{not json at all", "{not json at all"),
    ],
)
def test_render_memory_text(raw, expected):
    assert render_memory_text(raw) == expected


def test_render_drops_categories_noise():
    out = render_memory_text('{"preference":"Likes tea","categories":["a","b","c"]}')
    assert out == "Likes tea"
    assert "categories" not in out
