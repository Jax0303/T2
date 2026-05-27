"""Unit tests for the v4 robustness changes.

Covers four edits added on top of v3.1:

  Phase 0 - loader handles both `tables/{hmt,raw}/` and `tables/tables/{hmt,raw}/`
  Plan B - OriginalTable.resolve() fuzzy fallback (token/sequence-similarity)
  Plan C - symbolic_eval._safe_eval allows max/min/abs/int/round/sum only
  Plan D - RAGAgent.run() lowers w_verify for MULTI_OP_FORMULA / ARITHMETIC_AGG

These tests are intentionally offline: no model loads, no Chroma reads.
The `chromadb` import in `rag_agent.stores.vector_store` is satisfied by a
minimal stub installed before the package is imported.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub chromadb so the stores package can be imported without the real dep.
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = types.ModuleType("chromadb")
    sys.modules["chromadb.config"] = types.ModuleType("chromadb.config")


class TestLoaderNestedPaths(unittest.TestCase):
    """Phase 0: loader finds tables under either layout."""

    def _write_table(self, path: Path, table_id: str):
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{table_id}.json").write_text(
            json.dumps({"data": [[1, 2]], "top_root": {}, "left_root": {}}),
            encoding="utf-8",
        )

    def _make_root(self, base: Path) -> Path:
        """Create a root that _find_data_root accepts (needs train_samples.jsonl)."""
        (base / "data").mkdir(parents=True, exist_ok=True)
        (base / "data" / "train_samples.jsonl").write_text("", encoding="utf-8")
        return base

    def test_flat_layout(self):
        from rag_agent.data.loader import load_table
        with tempfile.TemporaryDirectory() as d:
            root = self._make_root(Path(d))
            self._write_table(root / "data" / "tables" / "hmt", "T1")
            tbl = load_table("T1", data_dir=str(root))
            self.assertIsNotNone(tbl)
            self.assertEqual(tbl["table_id"], "T1")

    def test_nested_layout(self):
        from rag_agent.data.loader import load_table
        with tempfile.TemporaryDirectory() as d:
            root = self._make_root(Path(d))
            self._write_table(root / "data" / "tables" / "tables" / "hmt", "T2")
            tbl = load_table("T2", data_dir=str(root))
            self.assertIsNotNone(tbl)
            self.assertEqual(tbl["table_id"], "T2")

    def test_missing_returns_none(self):
        from rag_agent.data.loader import load_table
        with tempfile.TemporaryDirectory() as d:
            root = self._make_root(Path(d))
            (root / "data" / "tables" / "hmt").mkdir(parents=True)
            self.assertIsNone(load_table("NOPE", data_dir=str(root)))


class TestFuzzyResolver(unittest.TestCase):
    """Plan B: fuzzy fallback only kicks in when exact match fails, and
    the fuzzy ordering is preserved (not overridden by path specificity)."""

    @classmethod
    def setUpClass(cls):
        from rag_agent.stores.original_store import OriginalTable
        cls.OriginalTable = OriginalTable

    def _table(self):
        OT = self.OriginalTable
        t = OT(
            table_id="dummy",
            title="Immigrants by region",
            data=[[100, 200], [150, 250], [80, 120], [330, 570]],
            top_paths=[["economic class"], ["family class"]],
            left_paths=[
                ["percent", "source region", "southern asia"],
                ["percent", "source region", "southeast asia"],
                ["percent", "source region", "east asia"],
                ["percent", "total"],
            ],
        )
        t.top_paths_by_col = {i: p for i, p in enumerate(t.top_paths)}
        t.left_paths_by_row = {i: p for i, p in enumerate(t.left_paths)}
        return t

    def test_exact_match_preferred(self):
        t = self._table()
        self.assertEqual(t.resolve("southern asia", "economic class"), (0, 0, 100))

    def test_fuzzy_hyphenated_token(self):
        # 'southern-asia' isn't word-boundary matched; fuzzy should still find row 0.
        t = self._table()
        self.assertEqual(t.resolve("southern-asia", "economic class"), (0, 0, 100))

    def test_fuzzy_picks_correct_row_not_longest_path(self):
        # The risk: 'southeast asia' has a longer header than 'southern asia',
        # so a naive specificity sort after fuzzy match would mis-pick it.
        t = self._table()
        r, c, v = t.resolve("southern-asia", "economic class")
        self.assertEqual((r, c), (0, 0))
        self.assertEqual(v, 100)

    def test_fuzzy_picks_southeast_correctly(self):
        t = self._table()
        self.assertEqual(t.resolve("southeast-asia", "economic class"), (1, 0, 150))

    def test_fuzzy_returns_none_for_unrelated(self):
        t = self._table()
        self.assertIsNone(t.resolve("north america", "economic class"))

    def test_blank_axis_returns_first_match(self):
        # When col_header is empty, all cols are candidates; row fuzzy still works.
        t = self._table()
        result = t.resolve("east asia", "")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 2)  # row index for 'east asia'


class TestSafeEvalFunctions(unittest.TestCase):
    """Plan C: AST evaluator whitelists max/min/abs/int/round/sum only."""

    def _eval(self, expr, env):
        from rag_agent.extract.symbolic_eval import _safe_eval
        return _safe_eval(expr, env)

    def test_max_min(self):
        env = {"x1": 5.0, "x2": 9.0, "x3": 3.0}
        self.assertEqual(self._eval("max(x1, x2, x3)", env), 9.0)
        self.assertEqual(self._eval("min(x1, x2, x3)", env), 3.0)

    def test_int_truncates_division(self):
        # =INT(G7/G8) pattern from arithmetic_agg.
        self.assertEqual(self._eval("int(x1 / x2)", {"x1": 7.0, "x2": 3.0}), 2.0)

    def test_abs_diff(self):
        self.assertEqual(self._eval("abs(x1 - x2)", {"x1": 3.0, "x2": 5.0}), 2.0)

    def test_round(self):
        self.assertEqual(self._eval("round(x1, 2)", {"x1": 1.2345}), 1.23)

    def test_arithmetic_still_works(self):
        self.assertAlmostEqual(
            self._eval("(x1 + x2) / x3", {"x1": 1.0, "x2": 2.0, "x3": 4.0}),
            0.75,
        )

    def test_unsafe_calls_blocked(self):
        for bad in ['__import__("os")', 'open("x")', "globals()", "foo(1)"]:
            with self.subTest(expr=bad):
                with self.assertRaises(ValueError):
                    self._eval(bad, {})

    def test_attribute_access_blocked(self):
        with self.assertRaises(ValueError):
            self._eval("x1.__class__", {"x1": 1.0})


class _StubLLM:
    name = "stub"
    def complete(self, system, user, max_tokens=256):
        return '{"cells": [], "expression": ""}'


class _StubVectorStore:
    """Returns a single fake hit, so RAGAgent.run() reaches the rerank call."""
    def search(self, query, top_k_vectors, top_k_tables):
        from rag_agent.stores.vector_store import VectorHit
        return [VectorHit(table_id="T_only", score=0.5, vector_id="v1",
                          chunk_text="", meta={})]


class TestVerifierWeightsByClass(unittest.TestCase):
    """Plan D: w_verify is lowered for multi-cell arithmetic queries.

    We monkeypatch `rerank` inside the agent module to capture the
    (w_vector, w_verify) it was called with, then invoke the agent with
    queries that classify to MULTI_OP_FORMULA, ARITHMETIC_AGG, and a
    non-arithmetic lookup respectively.
    """

    def _run_with_query(self, query: str):
        from rag_agent import agent as agent_mod
        from rag_agent.stores.original_store import OriginalStore, OriginalTable
        from rag_agent.agent import RAGAgent

        # Minimal OriginalStore with one table so rerank actually has something.
        store = OriginalStore()
        t = OriginalTable(
            table_id="T_only", title="x",
            data=[[1.0]], top_paths=[["c"]], left_paths=[["r"]],
        )
        t.top_paths_by_col = {0: ["c"]}
        t.left_paths_by_row = {0: ["r"]}
        store._tables["T_only"] = t

        captured = {}
        original_rerank = agent_mod.rerank

        def spy(query, hits, original, w_vector=0.7, w_verify=0.3):
            captured["w_vector"] = w_vector
            captured["w_verify"] = w_verify
            return original_rerank(query, hits, original,
                                   w_vector=w_vector, w_verify=w_verify)

        agent_mod.rerank = spy
        try:
            ag = RAGAgent(store, _StubVectorStore(), llm=_StubLLM())
            ag.run(query)
        finally:
            agent_mod.rerank = original_rerank
        return captured

    def test_multi_op_formula_lowers_w_verify(self):
        # Two math symbols → MULTI_OP_FORMULA (strongest classifier signal).
        from rag_agent.router.query_classifier import classify_query, QueryType
        q = "what is (x1 + x2) / x3 in 2015?"
        self.assertEqual(classify_query(q).qtype, QueryType.MULTI_OP_FORMULA)
        captured = self._run_with_query(q)
        self.assertAlmostEqual(captured["w_vector"], 0.9)
        self.assertAlmostEqual(captured["w_verify"], 0.1)

    def test_arithmetic_agg_lowers_w_verify(self):
        # "ratio of X to Y" → ARITHMETIC_AGG.
        from rag_agent.router.query_classifier import classify_query, QueryType
        q = "what is the ratio of X to Y in 2015?"
        self.assertEqual(classify_query(q).qtype, QueryType.ARITHMETIC_AGG)
        captured = self._run_with_query(q)
        self.assertAlmostEqual(captured["w_vector"], 0.9)
        self.assertAlmostEqual(captured["w_verify"], 0.1)

    def test_default_lookup_keeps_w_verify(self):
        # Plain lookup question; should use default 0.7/0.3.
        captured = self._run_with_query("what was the value of X in 2015?")
        self.assertAlmostEqual(captured["w_vector"], 0.7)
        self.assertAlmostEqual(captured["w_verify"], 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
