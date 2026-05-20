"""Stage-routing policy.

For each QueryIntent the agent decides which stages to run. Skipped stages
are reported in the trace so we can verify the skip-decision numerically.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from .query_classifier import QueryIntent, QueryType


class Stage(str, Enum):
    RETRIEVE = "retrieve"          # vector search
    VERIFY = "verify"              # cross-check candidates vs OriginalStore
    SYMBOLIC = "symbolic"          # cell extract + pandas eval
    LLM_ANSWER = "llm_answer"      # fall back / general reader


@dataclass
class Plan:
    stages: List[Stage]
    reason: str


def plan_stages(intent: QueryIntent) -> Plan:
    if intent.qtype == QueryType.REASONING_ONLY:
        # No table needed — short-circuit straight to LLM.
        return Plan([Stage.LLM_ANSWER], "reasoning-only: skip retrieve+verify+symbolic")

    base = [Stage.RETRIEVE, Stage.VERIFY]

    if intent.needs_symbolic:
        # symbolic compute is primary; LLM still runs as fallback if extraction fails.
        return Plan(base + [Stage.SYMBOLIC, Stage.LLM_ANSWER],
                    f"{intent.qtype.value}: symbolic compute primary, LLM fallback")

    # lookup / single_arg / comparison: LLM reads the verified table directly.
    return Plan(base + [Stage.LLM_ANSWER],
                f"{intent.qtype.value}: retrieve+verify+LLM (no symbolic)")
