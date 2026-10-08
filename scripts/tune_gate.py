"""Tune the gate question.

The whole premise of the PoC is that a decision model can tell "Hello" from "I want
a drink". That is a claim about a *specific question wording*, not about the model,
so it needs measuring rather than assuming.

This asks every candidate wording about every labelled utterance. All candidates go
in a single call per utterance -- extra questions about the same state are nearly
free, so the entire sweep costs one round trip per utterance rather than one per
(utterance, candidate).

Reported per candidate: the worst-case margin between the lowest positive and the
highest negative. A candidate only separates the classes if that margin is > 0.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

from decider_memory.deciders import make_decider, noul, score  # noqa: E402

load_dotenv()

# True  = answering this well genuinely depends on a stored personal fact
# False = it does not; a memory lookup is wasted latency and wasted context
LABELLED: list[tuple[str, bool]] = [
    ("I want a drink", True),
    ("Order me some lunch", True),
    ("Book me a flight to Toronto", True),
    ("What should I cook for dinner tonight?", True),
    ("Can you recommend a restaurant for tonight?", True),
    ("Where should I go on holiday this year?", True),
    ("Buy a birthday present for my dog", True),
    ("Review this function and match my usual style", True),
    ("Is there anything I should avoid on this menu?", True),
    ("Plan my morning for me", True),
    ("Hello", False),
    ("Thanks, that's great!", False),
    ("What is 2 + 2?", False),
    ("What's the capital of France?", False),
    ("Explain how TCP's three-way handshake works", False),
    ("ok", False),
    ("What time is it in Tokyo right now?", False),
    ("Summarise the text I just pasted", False),
    ("Who won the 1998 World Cup?", False),
    ("Convert 10 miles into kilometres", False),
]

CANDIDATES: dict[str, dict] = {
    # The naive wording, as a baseline.
    "plain": noul(
        "Answering this message well depends on recalling a stored personal fact "
        "or preference about this specific user."
    ),
    # Same idea, but with the boundary spelled out. The docs say criteria sharpen
    # the true/false split, and this is where that should show up.
    "criteria": noul(
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
    ),
    # Framed as a cost/benefit call rather than a fact about the message.
    "lookup": noul(
        "A lookup in this user's stored personal profile would change how this "
        "message should be answered.",
        criteria={
            "true": "Personal preferences would change the answer.",
            "false": "A lookup would return nothing useful; the answer is "
            "user-independent or purely social.",
        },
    ),
    # Inverted, to check the model is not just biased toward one end.
    "generic": noul(
        "This message can be answered perfectly well without knowing anything at "
        "all about who is asking.",
        criteria={
            "true": "Generic. Any user would get the same correct answer.",
            "false": "Personal. The answer depends on this user's own preferences "
            "or circumstances.",
        },
    ),
    # A graded version -- more headroom for picking a threshold.
    "graded": score(
        "How much does answering this message well depend on stored personal "
        "facts about this specific user?",
        [
            "not at all - purely social, or the same answer for everyone",
            "slightly - personalisation would be a minor nicety",
            "moderately - personal preferences would improve the answer",
            "heavily - the answer is wrong without this user's preferences",
        ],
    ),
}


def value_of(answer) -> float:
    """Collapse any answer type to a single comparable number."""
    if isinstance(answer, float):
        return answer
    if hasattr(answer, "fraction"):  # ScoreAnswer
        return answer.fraction
    return answer.confidence  # ChoiceAnswer


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "sagemaker"
    decider = make_decider(backend)

    def run(item: tuple[str, bool]) -> tuple[str, bool, dict[str, float]]:
        text, label = item
        answers = decider.ask(text, CANDIDATES)
        return text, label, {k: value_of(v) for k, v in answers.items()}

    # Concurrency is where this endpoint earns its keep: requests arriving together
    # are coalesced into one GPU pass.
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(run, LABELLED))

    names = list(CANDIDATES)
    print(f"\nbackend: {decider.name}\n")
    header = f"{'utterance':<46} {'want':<5} " + " ".join(f"{n:>9}" for n in names)
    print(header)
    print("-" * len(header))
    for text, label, vals in sorted(rows, key=lambda r: (not r[1], r[0])):
        cells = " ".join(f"{vals[n]:>9.3f}" for n in names)
        print(f"{text[:45]:<46} {str(label):<5} {cells}")

    # For each candidate: does a single threshold separate the two classes?
    print("\n--- separation ---")
    print(f"{'candidate':<12} {'min(pos)':>9} {'max(neg)':>9} {'margin':>8}  {'threshold':>9}")
    best: tuple[float, str, float] | None = None
    for n in names:
        pos = [v[n] for _, lab, v in rows if lab]
        neg = [v[n] for _, lab, v in rows if not lab]
        # "generic" is phrased the other way round, so flip it to compare fairly.
        if n == "generic":
            pos, neg = [1 - p for p in pos], [1 - q for q in neg]
        lo_pos, hi_neg = min(pos), max(neg)
        margin = lo_pos - hi_neg
        mid = (lo_pos + hi_neg) / 2
        flag = "  <-- separates" if margin > 0 else ""
        print(f"{n:<12} {lo_pos:>9.3f} {hi_neg:>9.3f} {margin:>8.3f}  {mid:>9.3f}{flag}")
        if best is None or margin > best[0]:
            best = (margin, n, mid)

    # A margin can be negative and the candidate still be usable, so also report
    # the best accuracy any threshold achieves.
    print("\n--- best achievable accuracy per candidate ---")
    for n in names:
        scored = [
            ((1 - v[n]) if n == "generic" else v[n], lab) for _, lab, v in rows
        ]
        cuts = sorted({s for s, _ in scored})
        accs = []
        for c in cuts:
            correct = sum(1 for s, lab in scored if (s >= c) == lab)
            accs.append((correct / len(scored), c))
        acc, cut = max(accs)
        print(f"{n:<12} accuracy {acc:>6.0%} at threshold {cut:.3f}")

    if best:
        print(f"\nwidest margin: {best[1]!r} (margin {best[0]:.3f}, threshold ~{best[2]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
