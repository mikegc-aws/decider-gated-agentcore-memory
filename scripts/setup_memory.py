"""Create the AgentCore Memory resource used by the proof of concept, and seed it.

Run once:

    .venv/bin/python scripts/setup_memory.py

Writes the resulting memory id to `.memory_id` so the demos can pick it up.
Seeding matters: long-term memory records only exist after the extraction
pipeline has chewed on some real conversation events, which takes a few minutes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType

REGION = "us-west-2"
MEMORY_NAME = "decider_gated_memory_poc"
# A deliberately fictional persona. Every "fact" below is invented demo data,
# chosen to span several unrelated areas of a life so that a semantic top-k search
# has a realistic chance of returning the wrong one.
ACTOR_ID = "demo_user"
ROOT = Path(__file__).resolve().parent.parent

# Seed conversations. Each is a separate session so that the user-preference and
# semantic strategies get a varied spread of facts to extract -- we want a memory
# store where a top-k search is *plausibly* wrong, not one with a single fact in it.
SEED_SESSIONS: dict[str, list[tuple[str, str]]] = {
    "seed_drinks": [
        ("I'm trying to cut down on caffeine, so decaf oat flat white is my go-to now.", "USER"),
        ("Noted -- decaf oat flat white it is.", "ASSISTANT"),
        ("And I really can't stand sparkling water. Still water only, please.", "USER"),
        ("Understood, still water rather than sparkling.", "ASSISTANT"),
    ],
    "seed_food": [
        ("I'm vegetarian, and I'm allergic to peanuts -- that one's serious.", "USER"),
        ("That's important, I'll keep peanuts away entirely.", "ASSISTANT"),
        ("My favourite cuisine is probably Vietnamese. I could eat pho every week.", "USER"),
        ("Vietnamese noted as your favourite.", "ASSISTANT"),
    ],
    "seed_travel": [
        ("I always book an aisle seat, I hate climbing over people.", "USER"),
        ("Aisle seat preference saved.", "ASSISTANT"),
        ("I live in Vancouver, so I usually fly out of YVR.", "USER"),
        ("Got it, Vancouver and YVR as your home airport.", "ASSISTANT"),
    ],
    "seed_work": [
        ("I write mostly Python, and I strongly prefer tabs-free code -- four spaces.", "USER"),
        ("Four spaces, no tabs. Noted.", "ASSISTANT"),
        ("I'm a backend engineer, mostly working on distributed systems.", "USER"),
        ("Thanks, that helps me pitch explanations at the right level.", "ASSISTANT"),
    ],
    "seed_misc": [
        ("I have a dog called Biscuit, a very opinionated cocker spaniel.", "USER"),
        ("Biscuit the cocker spaniel, noted.", "ASSISTANT"),
        ("I'm trying to get to the gym three mornings a week before work.", "USER"),
        ("Three gym mornings a week -- I'll keep that in mind.", "ASSISTANT"),
    ],
}


def main() -> int:
    client = MemoryClient(region_name=REGION)

    # Reuse the memory if a previous run already made it.
    existing = None
    for mem in client.gmcp_client.list_memories(maxResults=100).get("memories", []):
        if (mem.get("id") or "").startswith(MEMORY_NAME):
            existing = mem.get("id")
            break

    if existing:
        memory_id = existing
        print(f"Reusing existing memory: {memory_id}")
    else:
        print(f"Creating memory {MEMORY_NAME!r} (this takes a minute or two)...")
        memory = client.create_memory_and_wait(
            name=MEMORY_NAME,
            description="Decider-gated AgentCore Memory session manager PoC",
            strategies=[
                {
                    StrategyType.USER_PREFERENCE.value: {
                        "name": "UserPreferences",
                        "namespaces": ["/preferences/{actorId}"],
                    }
                },
                {
                    StrategyType.SEMANTIC.value: {
                        "name": "SemanticFacts",
                        "namespaces": ["/facts/{actorId}"],
                    }
                },
            ],
            event_expiry_days=90,
            max_wait=600,
            poll_interval=10,
        )
        memory_id = memory.get("id")
        print(f"Created memory: {memory_id}")

    (ROOT / ".memory_id").write_text(memory_id + "\n")

    # Record the strategy ids/namespaces -- the retrieval config needs the real
    # namespace paths, which include the generated strategy id for some strategies.
    strategies = client.get_memory_strategies(memory_id)
    info = [
        {
            "id": s.get("strategyId") or s.get("memoryStrategyId"),
            "name": s.get("name"),
            "type": s.get("type") or s.get("memoryStrategyType"),
            "namespaces": s.get("namespaces"),
        }
        for s in strategies
    ]
    (ROOT / ".strategies.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Strategies:")
    for s in info:
        print(f"  {s['type']:<16} {s['name']:<16} {s['namespaces']}")

    # Seed events so the extraction pipeline has something to build records from.
    print("\nSeeding conversation events...")
    for session_id, messages in SEED_SESSIONS.items():
        client.create_event(
            memory_id=memory_id,
            actor_id=ACTOR_ID,
            session_id=session_id,
            messages=messages,
        )
        print(f"  seeded {session_id} ({len(messages)} messages)")
        time.sleep(1)  # keep event timestamps distinct

    print(
        "\nDone. Long-term extraction runs asynchronously -- give it a few minutes, "
        "then check with:\n  .venv/bin/python scripts/check_memory.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
