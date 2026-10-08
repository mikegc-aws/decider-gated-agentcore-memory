"""A decider-gated fork of the AgentCore Memory session manager.

What upstream does
------------------
`AgentCoreMemorySessionManager` registers a `MessageAddedEvent` hook,
`retrieve_customer_context`. On **every** user text message it fans out over the
configured namespaces, runs a semantic `retrieve_memory_records` with a `topK`,
drops anything under a fixed `relevance_score`, and splices whatever survives into
the user's message as a `<user_context>` block.

Two things about that are blunt:

1. **It always fires.** "Hello" gets the same semantic search as "what should I
   drink?". That is a round trip to the memory service on the critical path of a
   turn where the result cannot possibly help.

2. **top-k returns k things whether or not k are relevant.** The only filter is a
   static score floor, and on real data the scores are packed far too tightly for
   one to work. Measured against the PoC's own memory store, a search for
   `"drink preference"` returns, in the same top-10, "the user's go-to drink is a
   decaf oat flat white" and -- only ~0.11 lower -- "the user writes mostly Python
   and prefers four spaces". No single floor keeps the first and drops the last.
   Everything above the default 0.2 lands in the prompt.

What this fork changes
----------------------
Exactly one method, `retrieve_customer_context`. Everything else -- writing events,
restoring sessions, batching, metadata, async mode -- is inherited untouched,
because the write path is not what is being questioned here.

    before gate   ->  is a lookup worth making at all?      1 noul question
    after lookup  ->  is each returned memory relevant?     N noul questions, 1 call

The second step batches deliberately. Extra questions about the same state are
nearly free, so validating ten memories is one round trip, not ten.

Subclassing rather than copying 1,300 lines is the point: the gate is a policy
change to a single decision, and it should be readable as one.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from strands.hooks import MessageAddedEvent

from .config import GATE_QUESTION, DeciderGateConfig, GateStats, validation_question
from .deciders import Decider
from .records import render_memory_text

logger = logging.getLogger(__name__)


class DeciderGatedAgentCoreMemorySessionManager(AgentCoreMemorySessionManager):
    """AgentCore Memory session manager with a decision model in front of retrieval.

    Args:
        agentcore_memory_config: Standard upstream config (memory id, actor,
            session, retrieval namespaces).
        decider: The decision model used for both the gate and the per-memory
            validation. Pass `None` to disable gating entirely, which makes this
            class behave exactly like upstream -- useful as the control arm of a
            comparison.
        gate_config: Thresholds and switches. See `DeciderGateConfig`.
        stats: Optional shared counters, so a caller can aggregate across turns.
    """

    def __init__(
        self,
        agentcore_memory_config: AgentCoreMemoryConfig,
        decider: Optional[Decider] = None,
        gate_config: Optional[DeciderGateConfig] = None,
        region_name: Optional[str] = None,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        stats: Optional[GateStats] = None,
        **kwargs: Any,
    ) -> None:
        self.decider = decider
        self.gate_config = gate_config or DeciderGateConfig()
        self.stats = stats if stats is not None else GateStats()
        super().__init__(
            agentcore_memory_config=agentcore_memory_config,
            region_name=region_name,
            boto_session=boto_session,
            boto_client_config=boto_client_config,
            **kwargs,
        )

    # ------------------------------------------------------------------ helpers

    def _trace(self, message: str) -> None:
        if self.gate_config.trace:
            print(f"    [memory] {message}")

    def _ask(self, state: Any, questions: dict[str, dict]) -> dict:
        """One decider round trip, with timing folded into the stats."""
        started = time.perf_counter()
        try:
            return self.decider.ask(state, questions)
        finally:
            self.stats.decider_calls += 1
            self.stats.decider_ms += (time.perf_counter() - started) * 1000

    def _should_retrieve(self, user_query: str) -> bool:
        """The gate: is a memory lookup worth making for this message at all?"""
        if not (self.gate_config.gate and self.decider):
            return True

        self.stats.gate_calls += 1
        try:
            answers = self._ask(user_query, {"worth": GATE_QUESTION})
        except Exception as e:
            # Fail open. A broken decider must not cost the agent its memory.
            self.stats.decider_errors += 1
            logger.warning("Gate decision failed, retrieving anyway: %s", e)
            self._trace(f"gate ERROR ({e}) -> retrieving anyway")
            return True

        p = float(answers["worth"])
        opened = p >= self.gate_config.gate_threshold
        if opened:
            self.stats.gate_opened += 1
        else:
            self.stats.gate_closed += 1
        verdict = "RETRIEVE" if opened else "SKIP"
        self._trace(
            f"gate p={p:.3f} (>= {self.gate_config.gate_threshold}) -> {verdict}"
        )
        return opened

    def _fetch_candidates(self, user_query: str) -> list[dict[str, Any]]:
        """Upstream's retrieval, but keeping scores and namespaces for the trace."""

        def retrieve_for_namespace(
            namespace: str, retrieval_config: RetrievalConfig
        ) -> list[dict[str, Any]]:
            resolved = namespace.format(
                actorId=self.config.actor_id,
                sessionId=self.config.session_id,
                memoryStrategyId=retrieval_config.strategy_id or "",
            )
            memories = self.memory_client.retrieve_memories(
                memory_id=self.config.memory_id,
                namespace_path=resolved,
                query=user_query,
                top_k=retrieval_config.top_k,
            )
            if retrieval_config.relevance_score:
                memories = [
                    m
                    for m in memories
                    if m.get("score", 0.0) >= retrieval_config.relevance_score
                ]
            out = []
            for memory in memories:
                if not isinstance(memory, dict):
                    continue
                content = memory.get("content", {})
                if not isinstance(content, dict):
                    continue
                raw = (content.get("text") or "").strip()
                if not raw:
                    continue
                # Flatten USER_PREFERENCE's JSON blobs to prose so every strategy
                # is judged on the same scale -- see records.render_memory_text.
                text = render_memory_text(raw)
                if text:
                    out.append(
                        {"text": text, "score": memory.get("score", 0.0), "namespace": resolved}
                    )
            return out

        candidates: list[dict[str, Any]] = []
        started = time.perf_counter()
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(retrieve_for_namespace, ns, rc): ns
                for ns, rc in self.config.retrieval_config.items()
            }
            for future in as_completed(futures):
                try:
                    candidates.extend(future.result())
                except Exception as e:
                    logger.error(
                        "Failed to retrieve memories for namespace %s: %s", futures[future], e
                    )
        self.stats.memory_ms += (time.perf_counter() - started) * 1000
        self.stats.retrieve_calls += 1
        return candidates

    def _validate(
        self, user_query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check every candidate against the request. One call per batch."""
        if not (self.gate_config.validate and self.decider) or not candidates:
            return candidates

        kept: list[dict[str, Any]] = []
        batch_size = self.gate_config.max_validate
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            questions = {f"m{i}": validation_question(c["text"]) for i, c in enumerate(batch)}
            try:
                answers = self._ask(user_query, questions)
            except Exception as e:
                # Fail open again: keep the batch rather than blind the agent.
                self.stats.decider_errors += 1
                logger.warning("Validation failed, keeping batch unfiltered: %s", e)
                self._trace(f"validate ERROR ({e}) -> keeping {len(batch)} unfiltered")
                kept.extend(batch)
                continue

            for i, candidate in enumerate(batch):
                p = float(answers.get(f"m{i}", 1.0))
                candidate["relevance"] = p
                if p >= self.gate_config.validate_threshold:
                    kept.append(candidate)
                    self._trace(f"  keep  p={p:.3f} s={candidate['score']:.3f} {candidate['text'][:72]}")
                else:
                    self.stats.memories_dropped += 1
                    self.stats.dropped_examples.append((user_query, candidate["text"], p))
                    self._trace(f"  DROP  p={p:.3f} s={candidate['score']:.3f} {candidate['text'][:72]}")
        return kept

    # ------------------------------------------------------- the overridden hook

    def retrieve_customer_context(self, event: MessageAddedEvent) -> None:
        """Gate, retrieve, validate, inject.

        Signature and injection format match upstream exactly, so this stays a
        drop-in replacement.
        """
        messages = event.agent.messages
        if not messages or messages[-1].get("role") != "user":
            return None
        content = messages[-1].get("content")
        if not content or "text" not in content[0]:
            return None
        if not self.config.retrieval_config:
            return None

        user_query = messages[-1]["content"][0]["text"]
        self.stats.turns += 1
        # What upstream would have done, for the comparison.
        self.stats.baseline_retrieve_calls += 1

        try:
            if not self._should_retrieve(user_query):
                return None

            candidates = self._fetch_candidates(user_query)
            self.stats.memories_returned += len(candidates)
            if not candidates:
                self._trace("no memories returned")
                return None

            kept = self._validate(user_query, candidates)
            self.stats.memories_kept += len(kept)
            if not kept:
                self._trace(f"all {len(candidates)} memories judged irrelevant -> nothing injected")
                return None

            # Prepend, exactly as upstream does: the user's own text stays last so
            # the model attends to the request rather than the context block.
            context_text = "\n".join(c["text"] for c in kept)
            self.stats.context_chars += len(context_text)
            tag = self.config.context_tag
            messages[-1]["content"].insert(
                0, {"text": f"<{tag}>{context_text}</{tag}>"}
            )
            self._trace(f"injected {len(kept)}/{len(candidates)} memories")

        except Exception as e:
            logger.error("Failed to retrieve customer context: %s", e)
