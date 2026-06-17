"""Tests for component 4: coverage check + fallback.

Coverage / decision logic is tested with hand-built results (no model). A
guarded HiTab test exercises the full retrieve -> coverage -> fallback path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from rag_agent.serialization import header_path as s2  # noqa: E402
from rag_agent.serialization.base import Chunk  # noqa: E402
from rag_agent.retrieve.hybrid_index import RetrievedChunk  # noqa: E402
from rag_agent.retrieve.operand_retrieval import Operand, OperandRetrievalResult  # noqa: E402
from rag_agent.fallback import (  # noqa: E402
    operand_coverage,
    decide_fallback,
    build_context,
    assemble_context,
    estimate_tokens,
)


class FakeTable:
    table_id = "t1"
    title = "Demo"
    _data = [[1234, 2345], [10, 20]]
    _cols = [["2023", "Q1"], ["2023", "Q2"]]
    _rows = [["Revenue"], ["Cost"]]

    @property
    def n_rows(self):
        return 2

    @property
    def n_cols(self):
        return 2

    def cell(self, r, c):
        return self._data[r][c]

    def col_path(self, c):
        return self._cols[c]

    def row_path(self, r):
        return self._rows[r]


def _rc(header_path, text="x", score=1.0):
    ch = Chunk(table_id="t1", chunk_id=f"c{id(header_path)}", text=text,
               scheme="S2", kind="cell", header_paths=[header_path])
    return RetrievedChunk(chunk=ch, score=score, bm25=0.0, dense=0.0)


def _result(operands, retrieved, confidence=0.9):
    ops = [Operand(header_path=p, row_path=p[:1], col_path=p[1:], query_text=" ".join(p))
           for p in operands]
    return OperandRetrievalResult(
        query="q", table_id="t1", operands=ops, per_operand=[],
        retrieved=retrieved, confidence=confidence,
    )


def test_coverage_full_when_all_operands_retrieved():
    res = _result([["Revenue", "2023", "Q1"]], [_rc(["Revenue", "2023", "Q1"])])
    cov = operand_coverage(res)
    assert cov.coverage_rate == 1.0
    assert cov.covered == [True]


def test_coverage_partial():
    res = _result(
        [["Revenue", "2023", "Q1"], ["Cost", "2023", "Q2"]],
        [_rc(["Revenue", "2023", "Q1"])],
    )
    cov = operand_coverage(res)
    assert cov.n_operands == 2 and cov.n_covered == 1
    assert cov.coverage_rate == 0.5


def test_decision_ok_when_high_coverage_and_confidence():
    res = _result([["Revenue", "2023", "Q1"]], [_rc(["Revenue", "2023", "Q1"])], confidence=0.9)
    dec = decide_fallback(res, coverage_threshold=0.7, confidence_threshold=0.3)
    assert dec.triggered is False
    assert dec.reason == "ok"


def test_decision_low_coverage_triggers():
    res = _result(
        [["Revenue", "2023", "Q1"], ["Cost", "2023", "Q2"]],
        [_rc(["Revenue", "2023", "Q1"])],
        confidence=0.9,
    )
    dec = decide_fallback(res, coverage_threshold=0.7, confidence_threshold=0.3)
    assert dec.triggered and dec.reason == "low_coverage"


def test_decision_low_confidence_triggers_even_if_covered():
    res = _result([["Revenue", "2023", "Q1"]], [_rc(["Revenue", "2023", "Q1"])], confidence=0.1)
    dec = decide_fallback(res, coverage_threshold=0.7, confidence_threshold=0.3)
    assert dec.triggered and dec.reason == "low_confidence"


def test_decision_no_operands():
    res = _result([], [], confidence=0.0)
    dec = decide_fallback(res)
    assert dec.triggered and dec.reason == "no_operands"


def test_build_context_uses_operand_cells_when_ok():
    res = _result([["Revenue", "2023", "Q1"]],
                  [_rc(["Revenue", "2023", "Q1"], text="Revenue > 2023 > Q1: 1234")],
                  confidence=0.9)
    dec = decide_fallback(res)
    ctx = build_context(res, FakeTable(), dec, max_tokens=4096)
    assert ctx.used_fallback is False
    assert "Revenue > 2023 > Q1: 1234" in ctx.text


def test_build_context_falls_back_to_whole_table():
    res = _result([["Revenue", "2023", "Q1"]], [_rc(["Revenue", "2023", "Q1"])], confidence=0.1)
    dec = decide_fallback(res)  # low_confidence
    ctx = build_context(res, FakeTable(), dec, max_tokens=4096)
    assert ctx.used_fallback is True
    # whole-table S2 rows -> both rows present
    assert "Revenue" in ctx.text and "Cost" in ctx.text


def test_fallback_respects_token_budget():
    res = _result([["Revenue", "2023", "Q1"]], [_rc(["Revenue", "2023", "Q1"])], confidence=0.1)
    dec = decide_fallback(res)
    ctx = build_context(res, FakeTable(), dec, max_tokens=1)  # absurdly small
    assert ctx.n_tokens <= max(1, estimate_tokens(ctx.text)) + 50
    assert ctx.truncated or ctx.n_chunks <= 1


def test_assemble_context_disable_fallback_arm():
    res = _result([["Revenue", "2023", "Q1"]], [_rc(["Revenue", "2023", "Q1"])], confidence=0.1)
    ctx = assemble_context(res, FakeTable(), enable_fallback=False)
    assert ctx.used_fallback is False
    assert ctx.decision.reason == "fallback_disabled"


def _hitab_available():
    from rag_agent.data.loader import _find_data_root
    try:
        _find_data_root("data/hitab")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hitab_available(), reason="HiTab data not downloaded")
def test_full_path_on_hitab_logs_decision():
    from rag_agent.data.loader import load_hitab
    from rag_agent.serialization import from_hitab_raw
    from rag_agent.retrieve.operand_retrieval import OperandTargetedRetriever
    from rag_agent.retrieve.encoders import HashingEncoder

    samples = load_hitab(split="dev", max_samples=10)
    r = OperandTargetedRetriever(encoder=HashingEncoder(dim=256), alpha=0.5)
    for s in samples:
        t = from_hitab_raw(s["table"])
        res = r.retrieve(s["question"], t, k=5)
        ctx = assemble_context(res, t, max_tokens=4096)
        d = ctx.to_dict()
        assert 0.0 <= d["decision"]["coverage_rate"] <= 1.0
        assert d["n_tokens"] <= 4096 + 50
        assert isinstance(d["used_fallback"], bool)
