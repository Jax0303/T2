# SPDX-License-Identifier: MIT
"""Unit tests for HPIR (header-path intent resolution).

These exercise the *deterministic* path only — no torch / chroma / HiTab data —
so they run in CI and in a bare container. The LLM-refined path is covered by a
fake-LLM stub.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.query import (  # noqa: E402
    expand_for_retrieval,
    extract_target_terms,
    resolve_against_table,
    resolve_intent,
)
from rag_agent.query.header_path_resolver import HeaderPathIntent  # noqa: E402
from rag_agent.stores.original_store import OriginalTable  # noqa: E402


def _toy_table() -> OriginalTable:
    """A 3×3 HiTab-style table with hierarchical top/left header paths."""
    top = {
        0: ["current $millions", "2013"],
        1: ["current $millions", "2014"],
        2: ["constant $millions", "2014"],
    }
    left = {
        0: ["all federal obligations"],
        1: ["department of defense"],
        2: ["health and human services"],
    }
    data = [
        [100.0, 110.0, 105.0],
        [40.0, 44.0, 42.0],
        [19.0, 20.0, 19.5],
    ]
    return OriginalTable(
        table_id="toy",
        title="federal s&e obligations to academic institutions",
        data=data,
        top_paths=[top[c] for c in range(3)],
        left_paths=[left[r] for r in range(3)],
        top_paths_by_col=top,
        left_paths_by_row=left,
    )


# --- term extraction -------------------------------------------------------

def test_extract_keeps_years_drops_cues_and_stopwords():
    q = "how many percent did federal s&e obligations increase between 2013 and 2014?"
    terms = extract_target_terms(q)
    assert "2013" in terms and "2014" in terms        # years are header signal
    assert "federal" in terms and "obligations" in terms
    for dropped in ("how", "many", "percent", "increase", "and", "did"):
        assert dropped not in terms


def test_extract_is_order_preserving_and_unique():
    terms = extract_target_terms("federal federal obligations obligations")
    assert terms == ["federal", "obligations"]


# --- retrieval expansion ---------------------------------------------------

def test_expansion_strips_narrative_keeps_targets():
    q = "how many percent did federal obligations increase between 2013 and 2014?"
    exp = expand_for_retrieval(q)
    assert "increase" not in exp and "how" not in exp
    assert "federal" in exp and "2014" in exp


def test_expansion_adds_operation_hint_for_ratio():
    # an arithmetic/ratio question should append a "total"/"percent" header hint
    exp = expand_for_retrieval("what percentage of theft cases involved a female accused?")
    assert "total" in exp.split() or "percent" in exp.split()


# --- table-grounded resolution --------------------------------------------

def test_resolve_binds_to_existing_paths_only():
    t = _toy_table()
    intent = resolve_against_table(
        "federal obligations in 2014", t, top_n_cols=3, top_n_rows=4
    )
    # every returned binding must be a real header path of the table
    real_cols = {" > ".join(t.col_path(c)) for c in range(t.n_cols)}
    real_rows = {" > ".join(t.row_path(r)) for r in range(t.n_rows)}
    for p in intent.col_paths:
        assert " > ".join(p) in real_cols
    for p in intent.row_paths:
        assert " > ".join(p) in real_rows


def test_resolve_picks_the_right_row_and_year_column():
    t = _toy_table()
    intent = resolve_against_table("federal obligations in 2014", t)
    assert ["all federal obligations"] in intent.row_paths
    assert any("2014" in p for p in intent.col_paths)


def test_resolve_sets_operation_and_symbolic_flag():
    t = _toy_table()
    intent = resolve_against_table(
        "how many percent did federal obligations increase from 2013 to 2014?", t
    )
    assert intent.needs_symbolic is True
    assert intent.operation in {"arithmetic_agg", "multi_op_formula"}


def test_binding_hint_is_readable():
    t = _toy_table()
    hint = resolve_against_table("federal obligations in 2014", t).binding_hint()
    assert "OPERATION:" in hint
    assert "all federal obligations" in hint


# --- LLM refinement (validated against inventory) --------------------------

class _FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload

    def complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        return self.payload


def test_llm_hallucinated_path_is_dropped():
    t = _toy_table()
    # LLM invents a column that does not exist; resolver must not surface it.
    bad = _FakeLLM('{"row_headers": ["all federal obligations"], '
                   '"col_headers": ["nonexistent header 9999"]}')
    intent = resolve_intent("federal obligations 2014", t, llm=bad)
    real_cols = {" > ".join(t.col_path(c)) for c in range(t.n_cols)}
    for p in intent.col_paths:
        assert " > ".join(p) in real_cols
    assert ["all federal obligations"] in intent.row_paths


def test_llm_none_equals_deterministic():
    t = _toy_table()
    a = resolve_against_table("federal obligations 2014", t)
    b = resolve_intent("federal obligations 2014", t, llm=None)
    assert a.row_paths == b.row_paths and a.col_paths == b.col_paths


def test_llm_garbage_output_falls_back():
    t = _toy_table()
    intent = resolve_intent("federal obligations 2014", t, llm=_FakeLLM("not json at all"))
    # falls back to deterministic, still grounded & non-empty
    assert isinstance(intent, HeaderPathIntent)
    assert intent.row_paths  # backfilled from deterministic ranking
