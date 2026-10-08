"""Show what is actually in long-term memory, and how flat the scores are.

Useful on its own: the output is the clearest evidence for why a static
`relevance_score` floor cannot work. Run a query and look at the score column --
relevant and irrelevant records land in the same narrow band.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bedrock_agentcore.memory import MemoryClient  # noqa: E402

from decider_memory.records import render_memory_text  # noqa: E402

REGION = "us-west-2"
ACTOR_ID = "demo_user"
NAMESPACES = ["/preferences/{actorId}", "/facts/{actorId}"]


def main() -> int:
    query = " ".join(sys.argv[1:]) or "drink preference"
    memory_file = ROOT / ".memory_id"
    if not memory_file.is_file():
        print("No .memory_id -- run scripts/setup_memory.py first.", file=sys.stderr)
        return 1
    memory_id = memory_file.read_text().strip()

    client = MemoryClient(region_name=REGION)
    print(f"memory: {memory_id}")
    print(f"query : {query!r}\n")

    all_scores: list[float] = []
    for namespace in NAMESPACES:
        resolved = namespace.format(actorId=ACTOR_ID)
        records = client.retrieve_memories(
            memory_id=memory_id, namespace_path=resolved, query=query, top_k=10
        )
        print(f"--- {resolved}  ({len(records)} records)")
        for record in records:
            score = record.get("score", 0.0)
            all_scores.append(score)
            text = render_memory_text((record.get("content") or {}).get("text", ""))
            print(f"  {score:.3f}  {text[:92]}")
        print()

    if all_scores:
        lo, hi = min(all_scores), max(all_scores)
        print(
            f"score range across {len(all_scores)} records: {lo:.3f} .. {hi:.3f} "
            f"(spread {hi - lo:.3f})"
        )
        print(
            "A single relevance_score floor has to separate the relevant records "
            "from the irrelevant ones inside that spread. That is the problem the "
            "decider is there to solve."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
