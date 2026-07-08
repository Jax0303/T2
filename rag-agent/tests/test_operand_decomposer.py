# SPDX-License-Identifier: MIT
"""Data-free unit tests for operand decomposition + the ceiling metric."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.schema import BenchTable, GoldOperand
from rag_agent.query.operand_decomposer import (
    candidate_paths, rank_paths, header_path_match_accuracy, fuzzy_score,
)


def _toy() -> BenchTable:
    return BenchTable(
        table_id="t1", title="Holdings",
        data=[[10, 11], [20, 21]],
        top_paths=[["year", "2022"], ["year", "2023"]],
        left_paths=[["assets", "cash"], ["assets", "bonds"]],
        source="toy",
    )


def test_candidate_paths_are_distinct_full_paths():
    paths = {p for p, _, _ in candidate_paths(_toy())}
    assert "assets > cash > year > 2022" in paths
    assert len(paths) == 4  # 2 rows x 2 cols, all distinct


def test_fuzzy_prefers_overlapping_path():
    a = fuzzy_score(["cash", "2023"], "assets > cash > year > 2023")
    b = fuzzy_score(["cash", "2023"], "assets > bonds > year > 2022")
    assert a > b


def test_rank_puts_relevant_path_first():
    ranked = rank_paths("what were cash holdings in 2023?", _toy(), matcher="fuzzy")
    assert ranked[0][0] == "assets > cash > year > 2023"


def test_ceiling_metric_perfect_and_none():
    t = _toy()
    gold = [GoldOperand(row=0, col=1, header_path=["assets", "cash", "year", "2023"], value=11)]
    acc = header_path_match_accuracy("cash holdings in 2023", t, gold, matcher="fuzzy")
    assert acc == 1.0
    # no gold operands -> excluded (None)
    assert header_path_match_accuracy("anything", t, [], matcher="fuzzy") is None
