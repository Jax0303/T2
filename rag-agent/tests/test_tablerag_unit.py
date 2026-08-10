# SPDX-License-Identifier: MIT
"""Pins the TableRAG index unit to what its source actually builds.

If these fail, "TableRAG baseline" stops being a true statement about this code.
Reference: google-research/table_rag/agent/retriever.py :: build_cell_corpus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench.schema import BenchTable
from rag_agent.serialization import tablerag_unit as trag


def _table():
    """2 data rows: one text stub column, two numeric columns."""
    return BenchTable(
        table_id="t1", title="crops",
        data=[["garlic", "1290", "2207"], ["kale", "92", "448"]],
        top_paths=[["item"], ["area", "2011"], ["area", "2016"]],
        left_paths=[["garlic"], ["kale"]],
    )


def test_numeric_columns_collapse_to_one_summary_doc_each():
    chunks = trag.serialize(_table())
    schema = [c for c in chunks if c.metadata["tablerag_kind"] == "schema"]
    assert len(schema) == 2                       # the two numeric columns
    assert '"min": 92.0' in schema[0].text and '"max": 1290.0' in schema[0].text
    # and no individual number is retrievable
    assert not any("1290" in c.text and "cell_value" in c.text for c in chunks)


def test_categorical_cells_are_deduplicated_by_value():
    t = BenchTable(
        table_id="t2", title="",
        data=[["north"], ["north"], ["south"]],
        top_paths=[["region"]],
        left_paths=[["a"], ["b"], ["c"]],
    )
    texts = [c.text for c in trag.serialize(t)]
    assert texts.count('{"column_name": "region", "cell_value": "north"}') == 1
    assert len(texts) == 2                        # north, south — not three rows


def test_cell_doc_carries_no_row_identity():
    doc = trag.cell_doc(_table(), 0, 0)
    assert doc == '{"column_name": "item", "cell_value": "garlic"}'
    assert "garlic" in doc          # the value happens to BE the row label here
    # but for a data column the row it came from is simply absent:
    t = _table()
    assert "garlic" not in trag.schema_doc(t, 1)


def test_column_name_mode_path_is_the_stronger_variant():
    leaf = trag.schema_doc(_table(), 1, "leaf")
    path = trag.schema_doc(_table(), 1, "path")
    assert '"column_name": "2011"' in leaf
    assert '"column_name": "area > 2011"' in path


def test_budget_caps_the_categorical_docs():
    t = BenchTable(
        table_id="t3", title="",
        data=[[f"v{i}"] for i in range(50)],
        top_paths=[["c"]],
        left_paths=[[f"r{i}"] for i in range(50)],
    )
    assert len(trag.serialize(t, max_encode_cell=10)) == 10
