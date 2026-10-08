"""Configuration for the decider-gated session manager."""

from __future__ import annotations

from dataclasses import dataclass, field

from .deciders import noul

# ---------------------------------------------------------------- the questions
#
# Both of these were tuned against a labelled set rather than guessed -- see
# scripts/tune_gate.py. The headline finding: the *wording* carries almost all of
# the quality. The obvious phrasing ("answering this depends on a stored fact
# about the user") scored 65% on strands-decider-2b, barely better than a coin
# flip. The same question with the true/false boundary spelled out below scored
# 100% with a 0.199 margin. If you change this text, re-run the tuner.

GATE_QUESTION = noul(
    "Answering this message well requires knowing a stored personal fact, "
    "preference, or history about this specific user.",
    criteria={
        "true": "The right answer differs from user to user. It depends on their "
        "tastes, restrictions, possessions, location, habits or past choices. "
        "Examples: what to eat or drink, what to buy, where to travel, "
        "anything phrased as 'my' or 'for me'.",
        "false": "The right answer is the same for everybody, or there is no "
        "question at all. Greetings, thanks, acknowledgements, arithmetic, "
        "general knowledge, definitions, unit conversion, or operating purely "
        "on text the user just supplied.",
    },
)


def validation_question(memory_text: str) -> dict:
    """Build the per-memory relevance question.

    The user's message goes in the shared `state`; the memory under test goes in
    the instructions. That split is what lets every candidate memory be checked in
    a single round trip.

    Tuned in scripts/tune_validator.py over 4 queries x ~9 memories. This wording
    reaches per-query AUC 1.000 and precision@k 100%, separating relevant from
    irrelevant at an absolute threshold of 0.379. The first draft of this question
    -- the same idea without the explicit true/false criteria -- managed AUC 0.812
    and dropped "go-to drink is a decaf oat flat white" for the query "I'd like a
    drink". The criteria block is doing the work.
    """
    return noul(
        f"Stored note about the user: «{memory_text}»\n\n"
        "This note is useful for answering the user's message in the state.",
        criteria={
            "true": "The note is about the same subject as the request, and an "
            "assistant answering it would want to know this.",
            "false": "The note is about an unrelated part of the user's life. "
            "True about them, but irrelevant to this request, and including it "
            "would only be a distraction.",
        },
    )


# Thresholds are per-backend, because the two models put their decision boundary
# at quite different absolute levels even on identical questions.
#
# A caveat worth knowing before trusting these numbers: across repeated tuning
# runs, *rankings* were stable for both models (per-query AUC came out 1.000 every
# time) but Jev's absolute values moved enough to shift its best pooled threshold
# noticeably, while strands-decider-2b's stayed put. So 2b is the tuned, primary
# path here; the Jev numbers are a reasonable starting point rather than a
# measured optimum, and anything load-bearing should re-run the tuners.
BACKEND_DEFAULTS: dict[str, dict[str, float]] = {
    "strands-decider-2b": {"gate_threshold": 0.19, "validate_threshold": 0.36},
    "jev": {"gate_threshold": 0.43, "validate_threshold": 0.50},
}


@dataclass
class DeciderGateConfig:
    """How the decision model is allowed to interfere with memory retrieval.

    Attributes:
        gate: Ask the decider whether a retrieval is worth making at all, before
            any call to the memory service. This is the latency/cost win.
        validate: Check each retrieved memory against the actual request before it
            reaches the agent's context. This is the precision win.
        gate_threshold: Minimum P(worth retrieving) to let a retrieval proceed.
            0.19 sits in the measured gap between the two classes for
            strands-decider-2b; Jev separates higher, around 0.43.
        validate_threshold: Minimum P(this memory is relevant) to keep a memory.
            0.36 is the midpoint of the measured gap for strands-decider-2b
            (relevant memories bottom out at 0.379, irrelevant ones top out at
            0.338). The band is narrow in absolute terms but the *ranking* is
            perfect, so err low: a threshold that is slightly too generous keeps
            a marginal memory, while one slightly too strict discards the single
            most useful fact. Jev scores higher and wants roughly 0.55.
        max_validate: Cap on memories checked in one call. The decider's context
            window is 3,072 tokens and long states are truncated, so a very large
            candidate set is split across calls rather than silently cut.
        trace: Print what the gate and validator decided. Noisy, but it is the
            whole point of a proof of concept.
    """

    gate: bool = True
    validate: bool = True
    gate_threshold: float = 0.19
    validate_threshold: float = 0.36
    max_validate: int = 12
    trace: bool = False

    @classmethod
    def for_backend(cls, decider_name: str, **overrides: object) -> "DeciderGateConfig":
        """Build a config with the measured thresholds for a given backend."""
        params = dict(BACKEND_DEFAULTS.get(decider_name, {}))
        params.update(overrides)  # type: ignore[arg-type]
        return cls(**params)  # type: ignore[arg-type]


@dataclass
class GateStats:
    """Counters so the demo can show what the gate actually bought.

    `baseline_*` figures are what the unmodified session manager would have done,
    which is what makes the before/after comparison honest.
    """

    turns: int = 0
    gate_calls: int = 0
    gate_opened: int = 0
    gate_closed: int = 0
    retrieve_calls: int = 0  # actual calls made to the memory service
    baseline_retrieve_calls: int = 0  # what upstream would have made
    memories_returned: int = 0
    memories_kept: int = 0
    memories_dropped: int = 0
    decider_calls: int = 0
    decider_ms: float = 0.0
    memory_ms: float = 0.0
    context_chars: int = 0  # characters of memory text actually injected
    # Fail-open events. These must be reported: because a broken decider
    # degrades silently to upstream behaviour, a whole run can look merely
    # "ineffective" when in fact every decision errored.
    decider_errors: int = 0
    dropped_examples: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def retrieval_calls_saved(self) -> int:
        return self.baseline_retrieve_calls - self.retrieve_calls

    def report(self) -> str:
        lines = [
            f"  turns                      {self.turns}",
            f"  gate decisions             {self.gate_calls}"
            f"  (open {self.gate_opened}, closed {self.gate_closed})",
            f"  memory retrievals made     {self.retrieve_calls}"
            f"  (baseline would be {self.baseline_retrieve_calls},"
            f" saved {self.retrieval_calls_saved})",
            f"  memories returned          {self.memories_returned}",
            f"  memories kept              {self.memories_kept}",
            f"  memories dropped           {self.memories_dropped}",
            f"  decider calls              {self.decider_calls}"
            f"  ({self.decider_ms:.0f} ms total)",
            f"  memory service time        {self.memory_ms:.0f} ms total",
            f"  context injected           {self.context_chars} chars"
            f"  (~{self.context_chars // 4} tokens)",
        ]
        if self.memories_returned:
            precision = self.memories_kept / self.memories_returned
            lines.append(f"  share of returned kept     {precision:.0%}")
        if self.decider_errors:
            lines.append(
                f"  !! decider errors          {self.decider_errors}"
                "  (failed open -- results below reflect UNGATED behaviour)"
            )
        return "\n".join(lines)
