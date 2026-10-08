"""Tune the per-memory validation question.

The first run of the demo exposed a real problem: with the obvious wording and an
absolute 0.5 cut, the validator threw away "the user's go-to drink is a decaf oat
flat white" (p=0.484) in response to "I'd like a drink, what should I get?". The
*ranking* was right -- that was the top-scoring memory for that query -- but the
absolute values all sit in a narrow band, so an absolute threshold cuts in the
wrong place.

So this measures two different things, because they fail independently:

  separation  can one absolute threshold split relevant from irrelevant, pooled
              across all queries? (what the session manager did originally)
  ranking     within a single query, do the relevant memories outrank the
              irrelevant ones? (ROC AUC per query, and precision@k)

If ranking is good but separation is poor, the fix is a relative rule -- keep what
is close to the best memory for *this* query -- not a better absolute number.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

from decider_memory.deciders import make_decider, noul, score  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# The real memory records this PoC seeds, keyed short for readability.
MEMORIES = {
    "coffee": "The user's go-to drink is a decaf oat flat white.",
    "water": "The user dislikes sparkling water and prefers still water only.",
    "caffeine": "The user is trying to cut down on caffeine.",
    "veg": "The user is vegetarian.",
    "peanut": "The user is allergic to peanuts (serious allergy).",
    "viet": "The user's favourite cuisine is Vietnamese.",
    "pho": "The user enjoys pho and could eat it every week.",
    "aisle": "The user always books an aisle seat when flying.",
    "yvr": "The user lives in Vancouver and usually flies out of YVR.",
    "python": "The user writes mostly Python and prefers four spaces, no tabs.",
    "job": "The user is a backend engineer working on distributed systems.",
    "dog": "The user has a dog called Biscuit, a very opinionated cocker spaniel.",
    "gym": "The user is trying to get to the gym three mornings a week.",
}

# (query, {memory key: is it genuinely useful here})
# Only the clear-cut calls are labelled; the genuinely arguable ones are left out
# rather than fudged, so the numbers below mean something.
CASES: list[tuple[str, dict[str, bool]]] = [
    (
        "I'd like a drink, what should I get?",
        {
            "coffee": True, "water": True, "caffeine": True,
            "python": False, "job": False, "dog": False, "gym": False,
            "aisle": False, "yvr": False,
        },
    ),
    (
        "What should I have for dinner?",
        {
            "veg": True, "peanut": True, "viet": True, "pho": True,
            "python": False, "job": False, "dog": False,
            "aisle": False, "yvr": False,
        },
    ),
    (
        "I'm booking a flight next week, anything you should know?",
        {
            "aisle": True, "yvr": True,
            "coffee": False, "python": False, "job": False, "dog": False,
            "gym": False, "pho": False,
        },
    ),
    (
        "Can you review this function for me?",
        {
            "python": True, "job": True,
            "coffee": False, "veg": False, "dog": False, "aisle": False,
            "yvr": False, "pho": False, "gym": False,
        },
    ),
]

CANDIDATES: dict[str, callable] = {
    # What the first demo run used.
    "plain": lambda t: noul(
        f"Here is a stored note about the user:\n\n«{t}»\n\n"
        "This note is genuinely useful for answering the user's message shown in "
        "the state."
    ),
    # Same, with the boundary spelled out -- the trick that fixed the gate.
    "criteria": lambda t: noul(
        f"Stored note about the user: «{t}»\n\n"
        "This note is useful for answering the user's message in the state.",
        criteria={
            "true": "The note is about the same subject as the request, and an "
            "assistant answering it would want to know this.",
            "false": "The note is about an unrelated part of the user's life. "
            "True about them, but irrelevant to this request, and including it "
            "would only be a distraction.",
        },
    ),
    # Framed as topical overlap, which is a narrower and more concrete judgement.
    "topic": lambda t: noul(
        f"Stored note: «{t}»\n\nThe note and the user's message in the state are "
        "about the same topic.",
        criteria={
            "true": "Same domain -- e.g. both about food, both about travel, "
            "both about code, both about drinks.",
            "false": "Different domains -- e.g. the note is about code and the "
            "message is about food.",
        },
    ),
    # Would omitting it hurt? A sharper counterfactual than "is it useful".
    "omit": lambda t: noul(
        f"Stored note: «{t}»\n\nAn assistant that did NOT know this note would "
        "give a noticeably worse answer to the user's message in the state.",
        criteria={
            "true": "Omitting it would make the answer generic, or wrong for this "
            "user.",
            "false": "Omitting it would change nothing about the answer.",
        },
    ),
    # Graded, for threshold headroom.
    "graded": lambda t: score(
        f"Stored note about the user: «{t}»\n\nHow relevant is this note to the "
        "user's message in the state?",
        [
            "irrelevant - a different part of their life entirely",
            "tangential - same person, unrelated need",
            "related - touches on the request",
            "directly on point - the answer should be built around it",
        ],
    ),
}


def value_of(answer) -> float:
    if isinstance(answer, float):
        return answer
    if hasattr(answer, "fraction"):
        return answer.fraction
    return answer.confidence


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random relevant memory outranks a random irrelevant one."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p, n in product(pos, neg))
    return wins / (len(pos) * len(neg))


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "sagemaker"
    decider = make_decider(backend)

    # One call per (case, candidate): all memories for that case go in together.
    jobs = [(qi, cname) for qi in range(len(CASES)) for cname in CANDIDATES]

    def run(job):
        qi, cname = job
        query, labels = CASES[qi]
        keys = list(labels)
        questions = {k: CANDIDATES[cname](MEMORIES[k]) for k in keys}
        answers = decider.ask(query, questions)
        return qi, cname, {k: value_of(answers[k]) for k in keys}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, jobs))

    scores: dict[tuple[int, str], dict[str, float]] = {
        (qi, cname): vals for qi, cname, vals in results
    }

    print(f"\nbackend: {decider.name}")

    for cname in CANDIDATES:
        print(f"\n{'=' * 78}\ncandidate: {cname}\n{'=' * 78}")
        all_pos: list[float] = []
        all_neg: list[float] = []
        aucs: list[float] = []
        p_at_k: list[float] = []

        for qi, (query, labels) in enumerate(CASES):
            vals = scores[(qi, cname)]
            pos = [vals[k] for k, lab in labels.items() if lab]
            neg = [vals[k] for k, lab in labels.items() if not lab]
            all_pos += pos
            all_neg += neg
            a = auc(pos, neg)
            aucs.append(a)

            # precision@k where k = number of truly relevant memories
            k = len(pos)
            ranked = sorted(labels, key=lambda m: vals[m], reverse=True)
            hits = sum(1 for m in ranked[:k] if labels[m])
            p_at_k.append(hits / k)

            print(f"\n  {query}")
            print(f"    AUC {a:.2f}   precision@{k} {hits}/{k}")
            for m in ranked:
                mark = "REL " if labels[m] else "  . "
                print(f"      {mark} {vals[m]:.3f}  {m:<9} {MEMORIES[m][:58]}")

        margin = min(all_pos) - max(all_neg)
        best_acc, best_cut = max(
            (
                (
                    sum(
                        1
                        for s, lab in [(p, True) for p in all_pos] + [(n, False) for n in all_neg]
                        if (s >= c) == lab
                    )
                    / (len(all_pos) + len(all_neg)),
                    c,
                )
                for c in sorted(set(all_pos + all_neg))
            )
        )
        mean_auc = sum(aucs) / len(aucs)
        mean_pk = sum(p_at_k) / len(p_at_k)
        print(f"\n  --- {cname} overall ---")
        print(f"    pooled min(rel) {min(all_pos):.3f}  max(irrel) {max(all_neg):.3f}  margin {margin:+.3f}")
        print(f"    best absolute threshold {best_cut:.3f} -> accuracy {best_acc:.0%}")
        print(f"    mean per-query AUC {mean_auc:.3f}   mean precision@k {mean_pk:.0%}")

    print(f"\n{'=' * 78}\nsummary\n{'=' * 78}")
    print(f"{'candidate':<10} {'margin':>8} {'bestAcc':>8} {'meanAUC':>8} {'meanP@k':>8}")
    for cname in CANDIDATES:
        all_pos, all_neg, aucs, pks = [], [], [], []
        for qi, (_, labels) in enumerate(CASES):
            vals = scores[(qi, cname)]
            pos = [vals[k] for k, lab in labels.items() if lab]
            neg = [vals[k] for k, lab in labels.items() if not lab]
            all_pos += pos
            all_neg += neg
            aucs.append(auc(pos, neg))
            k = len(pos)
            ranked = sorted(labels, key=lambda m: vals[m], reverse=True)
            pks.append(sum(1 for m in ranked[:k] if labels[m]) / k)
        pooled = [(p, True) for p in all_pos] + [(n, False) for n in all_neg]
        best_acc = max(
            sum(1 for s, lab in pooled if (s >= c) == lab) / len(pooled)
            for c in sorted(set(all_pos + all_neg))
        )
        print(
            f"{cname:<10} {min(all_pos) - max(all_neg):>+8.3f} {best_acc:>8.0%} "
            f"{sum(aucs) / len(aucs):>8.3f} {sum(pks) / len(pks):>8.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
