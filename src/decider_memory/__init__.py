"""A decider-gated fork of the Strands AgentCore Memory session manager.

Puts a small decision model (strands-decider-2b, or Jev) in front of long-term
memory retrieval, to answer two questions the upstream integration never asks:

    1. Is a memory lookup worth making for this message at all?
    2. Is each memory that came back actually relevant to what was asked?
"""

from .config import DeciderGateConfig, GateStats
from .deciders import (
    ChoiceAnswer,
    Decider,
    DeciderError,
    JevDecider,
    SageMakerDecider,
    ScoreAnswer,
    choice,
    make_decider,
    noul,
    score,
)
from .session_manager import DeciderGatedAgentCoreMemorySessionManager

__all__ = [
    "ChoiceAnswer",
    "Decider",
    "DeciderError",
    "DeciderGateConfig",
    "DeciderGatedAgentCoreMemorySessionManager",
    "GateStats",
    "JevDecider",
    "SageMakerDecider",
    "ScoreAnswer",
    "choice",
    "make_decider",
    "noul",
    "score",
]
