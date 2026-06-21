# SPDX-License-Identifier: MIT
"""Data-free unit tests for coverage assessment + fallback."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.schema import BenchTable, Chunk
from rag_agent.query.operand_decomposer import Operand
from rag_agent.retrieve.coverage import assess, apply_fallback


def _toy() -> BenchTable:
    return BenchTable(
        table_id="t1", title="Holdings",
        data=[[10, 11], [20, 21]],
        top_paths=[["year", "2022"], ["year", "2023"]],
        left_paths=[["assets", "cash"], ["assets", "bonds"]],
        source="toy",
    )


def test_high_confidence_no_fallback():
    ops = [Operand(["assets", "cash"], score=0.9), Operand(["assets", "bonds"], score=0.8)]
    rep = assess(ops, [], _toy(), tau_cov=0.7, score_floor=0.3)
    assert rep.coverage_rate == 1.0 and rep.fallback is False and rep.reason == "ok"


def test_low_confidence_triggers_fallback():
    ops = [Operand(["a"], score=0.1), Operand(["b"], score=0.2)]
    rep = assess(ops, [], _toy(), tau_cov=0.7, score_floor=0.3)
    assert rep.coverage_rate == 0.0 and rep.fallback is True
    assert "coverage<0.7" in rep.reason


def test_no_operands_triggers_fallback():
    rep = assess([], [], _toy(), tau_cov=0.7)
    assert rep.fallback is True and "no_operands" in rep.reason


def test_apply_fallback_appends_full_table():
    t = _toy()
    retrieved = [Chunk(table_id="t1", chunk_id="t1#r0", text="...", rows=[0], cols=[0, 1])]
    rep = assess([Operand(["a"], score=0.0)], retrieved, t, tau_cov=0.7)
    ctx = apply_fallback(retrieved, t, rep)
    assert any(c.chunk_id.endswith("#full") for c in ctx)
    # full chunk covers the previously-missing row 1
    full = next(c for c in ctx if c.chunk_id.endswith("#full"))
    assert full.covers(1, 1)


def test_apply_fallback_noop_when_confident():
    t = _toy()
    retrieved = [Chunk(table_id="t1", chunk_id="t1#r0", text="...", rows=[0], cols=[0, 1])]
    rep = assess([Operand(["assets", "cash"], score=0.9)], retrieved, t, tau_cov=0.7)
    assert apply_fallback(retrieved, t, rep) == retrieved
