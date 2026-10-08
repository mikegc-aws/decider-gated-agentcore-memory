"""Side-by-side: the upstream session manager vs the decider-gated fork.

Runs the same conversation twice against the same AgentCore Memory store.

    arm A  baseline  decider=None -- upstream behaviour exactly
    arm B  gated     strands-decider-2b in front of retrieval

Usage:
    .venv/bin/python demo.py                 # strands-decider-2b
    .venv/bin/python demo.py --decider jev   # Jev via OpenRouter
    .venv/bin/python demo.py --quiet         # skip the per-turn trace
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from bedrock_agentcore.memory.integrations.strands.config import (  # noqa: E402
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from strands import Agent  # noqa: E402
from strands.models import BedrockModel  # noqa: E402

from decider_memory import (  # noqa: E402
    DeciderGateConfig,
    DeciderGatedAgentCoreMemorySessionManager,
    GateStats,
    make_decider,
)

REGION = "us-west-2"
ACTOR_ID = "demo_user"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# A mixed conversation: some turns genuinely need the user's stored preferences,
# some plainly do not. The point of the demo is that upstream cannot tell them
# apart and so pays for a semantic search on all of them.
TURNS = [
    "Hello!",
    "What is 17 times 23?",
    "I'd like a drink, what should I get?",
    "Thanks!",
    "Can you explain what a bloom filter is?",
    "I'm booking a flight next week, anything you should know?",
    "What should I have for dinner?",
    "ok cool",
]

SYSTEM_PROMPT = (
    "You are a concise personal assistant. If a <user_context> block appears in "
    "the user's message, it holds facts recalled from long-term memory -- use them "
    "naturally and never mention the block itself. Keep answers to two sentences."
)


def build_retrieval_config() -> dict[str, RetrievalConfig]:
    """The two namespaces seeded by scripts/setup_memory.py.

    top_k=10 and the default 0.2 score floor are upstream's own defaults, kept
    deliberately so the comparison is against stock behaviour.
    """
    return {
        "/preferences/{actorId}": RetrievalConfig(top_k=10, relevance_score=0.2),
        "/facts/{actorId}": RetrievalConfig(top_k=10, relevance_score=0.2),
    }


def run_arm(
    slug: str,
    label: str,
    memory_id: str,
    decider,
    gate_config: DeciderGateConfig,
    turns: list[str],
) -> tuple[GateStats, float, list[str]]:
    """Run the whole conversation through one configuration.

    `slug` becomes part of the session id, which the service constrains to
    [a-zA-Z0-9][a-zA-Z0-9-_]* -- so it is kept separate from the display label.
    """
    stats = GateStats()
    session_id = f"demo-{slug}-{uuid.uuid4().hex[:8]}"

    manager = DeciderGatedAgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=ACTOR_ID,
            retrieval_config=build_retrieval_config(),
        ),
        decider=decider,
        gate_config=gate_config,
        region_name=REGION,
        stats=stats,
    )

    agent = Agent(
        model=BedrockModel(model_id=MODEL_ID, region_name=REGION),
        system_prompt=SYSTEM_PROMPT,
        session_manager=manager,
        callback_handler=None,  # keep stdout clean; we print the replies ourselves
    )

    print(f"\n{'=' * 78}\n  ARM: {label}\n{'=' * 78}")
    replies: list[str] = []
    wall_start = time.perf_counter()
    for turn in turns:
        print(f"\n  user> {turn}")
        result = agent(turn)
        reply = str(result).strip().replace("\n", " ")
        replies.append(reply)
        print(f"  bot > {reply[:200]}")
    wall = time.perf_counter() - wall_start
    return stats, wall, replies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decider", default="sagemaker", choices=["sagemaker", "jev"])
    parser.add_argument("--quiet", action="store_true", help="hide the per-turn trace")
    parser.add_argument("--gate-threshold", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    for noisy in ("bedrock_agentcore", "strands", "botocore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    memory_file = ROOT / ".memory_id"
    if not memory_file.is_file():
        print("No .memory_id -- run scripts/setup_memory.py first.", file=sys.stderr)
        return 1
    memory_id = memory_file.read_text().strip()

    decider = make_decider(args.decider)
    # Thresholds differ per backend: both separate the classes cleanly, but at
    # different absolute levels (measured in scripts/tune_gate.py and
    # scripts/tune_validator.py).
    gate_cfg = DeciderGateConfig.for_backend(
        decider.name, gate=True, validate=True, trace=not args.quiet
    )
    if args.gate_threshold is not None:
        gate_cfg.gate_threshold = args.gate_threshold

    print(f"memory : {memory_id}")
    print(f"actor  : {ACTOR_ID}")
    print(
        f"decider: {decider.name} "
        f"(gate >= {gate_cfg.gate_threshold}, keep memory >= {gate_cfg.validate_threshold})"
    )

    # Arm A: decider=None means every decider branch is bypassed -- this is
    # upstream's retrieve_customer_context behaviour.
    baseline_stats, baseline_wall, _ = run_arm(
        "baseline",
        "A - baseline (upstream: always retrieve, keep all top-k)",
        memory_id,
        decider=None,
        gate_config=DeciderGateConfig(gate=False, validate=False, trace=not args.quiet),
        turns=TURNS,
    )

    gated_stats, gated_wall, _ = run_arm(
        "gated",
        f"B - gated ({decider.name}: gate + per-memory validation)",
        memory_id,
        decider=decider,
        gate_config=gate_cfg,
        turns=TURNS,
    )

    print(f"\n{'=' * 78}\n  RESULTS\n{'=' * 78}")
    print(f"\nA - baseline (upstream)\n{baseline_stats.report()}")
    print(f"  wall clock                 {baseline_wall:.1f} s")
    print(f"\nB - gated ({decider.name})\n{gated_stats.report()}")
    print(f"  wall clock                 {gated_wall:.1f} s")

    saved = gated_stats.retrieval_calls_saved
    print(f"\n{'-' * 78}")
    print(
        f"Retrievals avoided : {saved}/{gated_stats.baseline_retrieve_calls} turns "
        f"({saved / max(gated_stats.baseline_retrieve_calls, 1):.0%}) never touched "
        "the memory service."
    )
    if baseline_stats.memories_returned:
        print(
            f"Context injected   : baseline put {baseline_stats.memories_returned} memory "
            f"records into the prompt across the conversation; "
            f"gated put {gated_stats.memories_kept} "
            f"({gated_stats.memories_kept / baseline_stats.memories_returned:.0%})."
        )
    print(
        f"Decider overhead   : {gated_stats.decider_calls} calls, "
        f"{gated_stats.decider_ms:.0f} ms total "
        f"({gated_stats.decider_ms / max(gated_stats.decider_calls, 1):.0f} ms each)."
    )
    print(
        f"Memory service time: baseline {baseline_stats.memory_ms:.0f} ms, "
        f"gated {gated_stats.memory_ms:.0f} ms."
    )

    if gated_stats.dropped_examples:
        print(f"\n{'-' * 78}\nMemories the validator rejected (top-k returned them anyway):")
        seen: set[tuple[str, str]] = set()
        for query, text, p in gated_stats.dropped_examples:
            key = (query, text[:50])
            if key in seen:
                continue
            seen.add(key)
            print(f'  p={p:.3f}  "{query[:38]}"\n            -> {text[:96]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
