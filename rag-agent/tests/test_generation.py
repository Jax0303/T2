"""Tests for component 5: generation (direct / codegen) + scoring.

Deterministic via MockLLM and the real codegen subprocess (no API key).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.serialization.base import Chunk  # noqa: E402
from rag_agent.retrieve.hybrid_index import RetrievedChunk  # noqa: E402
from rag_agent.retrieve.operand_retrieval import Operand, OperandRetrievalResult  # noqa: E402
from rag_agent.fallback import build_context, decide_fallback  # noqa: E402
from rag_agent.generation import (  # noqa: E402
    Answerer,
    MockLLM,
    run_codegen,
    extract_code,
    score_answer,
)
from rag_agent.generation.prompts import cells_from_chunks  # noqa: E402


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


def _cell_chunk(path, value):
    return Chunk(
        table_id="t1",
        chunk_id=f"t1::S2::{'_'.join(path)}",
        text=f"Demo\n{' > '.join(path)}: {value}",
        scheme="S2",
        kind="cell",
        header_paths=[path],
    )


def _bundle(operands, chunks, confidence=0.9, max_tokens=4096):
    rcs = [RetrievedChunk(chunk=c, score=1.0, bm25=0.0, dense=0.0) for c in chunks]
    ops = [Operand(header_path=p, row_path=p[:1], col_path=p[1:], query_text=" ".join(p))
           for p in operands]
    res = OperandRetrievalResult(
        query="q", table_id="t1", operands=ops, per_operand=[],
        retrieved=rcs, confidence=confidence,
    )
    dec = decide_fallback(res)
    return build_context(res, FakeTable(), dec, max_tokens=max_tokens)


# --- codegen primitives ---------------------------------------------------

def test_extract_code_handles_fenced_and_raw():
    assert extract_code("```python\nx=1\n```") == "x=1"
    assert extract_code("```\ny=2\n```") == "y=2"
    assert extract_code("answer = 5") == "answer = 5"


def test_run_codegen_computes_over_cells():
    cells = [{"path": ["Revenue", "2023", "Q1"], "value": "1,234"}]
    code = "answer = float(CELLS[0]['value'].replace(',', ''))"
    res = run_codegen(code, cells)
    assert res.ok and float(res.value) == 1234.0


def test_run_codegen_sum_of_cells():
    cells = [{"path": ["a"], "value": "10"}, {"path": ["b"], "value": "20"}]
    code = "answer = sum(int(c['value']) for c in CELLS)"
    res = run_codegen(code, cells)
    assert res.ok and res.value == "30"


def test_run_codegen_reports_error():
    res = run_codegen("answer = 1/0", [{"path": [], "value": "x"}])
    assert not res.ok and "ZeroDivision" in res.error


def test_run_codegen_empty_code():
    assert run_codegen("", []).ok is False


# --- cells parsing --------------------------------------------------------

def test_cells_from_chunks_parses_value():
    chunks = [_cell_chunk(["Revenue", "2023", "Q1"], "1234")]
    cells = cells_from_chunks(chunks)
    assert cells == [{"path": ["Revenue", "2023", "Q1"], "value": "1234"}]


# --- Answerer -------------------------------------------------------------

def test_answer_direct_with_mock():
    bundle = _bundle([["Revenue", "2023", "Q1"]],
                     [_cell_chunk(["Revenue", "2023", "Q1"], "1234")])
    ans = Answerer(MockLLM()).answer("what is revenue Q1?", bundle, mode="direct")
    assert ans.mode == "direct"
    assert ans.answer == "1234"


def test_answer_codegen_with_mock_executes():
    bundle = _bundle([["Revenue", "2023", "Q1"]],
                     [_cell_chunk(["Revenue", "2023", "Q1"], "1,234")])
    ans = Answerer(MockLLM()).answer("revenue?", bundle, mode="codegen")
    assert ans.mode == "codegen"
    assert ans.exec_ok is True
    assert float(ans.answer) == 1234.0


def test_answer_invalid_mode():
    bundle = _bundle([["a"]], [_cell_chunk(["a"], "1")])
    try:
        Answerer(MockLLM()).answer("q", bundle, mode="bogus")
        assert False, "should raise"
    except ValueError:
        pass


# --- scoring --------------------------------------------------------------

def test_score_answer_numeric_and_exact():
    assert score_answer("1234", [1234.0]) == {"em": False, "nm": True}
    assert score_answer("quebec", ["quebec"]) == {"em": True, "nm": True}
    assert score_answer("999", [1234.0]) == {"em": False, "nm": False}
