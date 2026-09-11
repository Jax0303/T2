# SPDX-License-Identifier: MIT
"""Bugs found by comparing against pinned Google/Huawei source methods."""
import copy
from types import SimpleNamespace
import pytest

from scripts import retrieval_accuracy as ra
from rag_agent.eval.artifacts import Selection, evidence_fields, validate_retrieval
from analysis.validated_tables import validate_answers


class Table:
    data = [["10", "20"], ["30", "40"]]
    n_rows = n_cols = 2
    def row_path(self, i): return ["group", f"r{i}"]
    def col_path(self, j): return ["year", ["Revenue", "Cost"][j]]


def test_default_rowcol_keeps_selected_column_header_without_union_leak():
    covers = [{("T", 0, 0), ("T", 0, 1)}, {("T", 0, 0), ("T", 1, 0)}]
    selected = ra.rowcol_select([0, 1], covers, ["row", "column"], [True, False], 1, 1,
                                grid={"T": (Table(), "Title")})
    text = "\n".join(selected.context)
    assert "Revenue" in text and "r0" in text and "10" in text
    assert "Cost" not in text and "20" not in text and "30" not in text
    assert selected.cells == {("T", 0, 0)}
    assert "year > Revenue" not in text  # full hierarchy is an explicit variant


def test_rowcol_path_headers_are_explicit():
    units = ra.subtable_units({("T", 0, 0)}, {"T": (Table(), "Title")}, header_mode="path")
    assert "year > Revenue" in units[0]["text"]


def test_one_row_column_document_does_not_gain_row_stub():
    text = ra.line_text(Table(), [(0, 0)], "Title", "s3c", "values", include_row_label=False)
    assert text == "Title | 10"
    assert ra.line_text(Table(), [(0, 0)], "Title", "s3c", "values") == "Title | r0|10"


@pytest.mark.parametrize("coordinate", [("Other", 0, 0), ("T", -1, 0), ("T", True, 0), ("T", 0)])
def test_bad_gold_coordinates_cannot_silently_select_another_cell(coordinate):
    good = {("T", 0, 0)}
    selection = Selection(good, [{"text": "10", "cells": list(good)}], True)
    row = {"query_id": "q", "table_id": "T", "mode": "all", "m": 1,
           "correct": 1, "cells_in_context": 1, "gold_table_in_context": 1,
           **evidence_fields(selection, good)}
    row["gold_cells"] = [coordinate]
    with pytest.raises(ValueError):
        validate_retrieval(row)


def test_same_id_different_question_is_not_comparable():
    r = {"query_id": "q", "question": "revenue?", "answer": [10], "mode": "all", "correct": 1}
    a = {"query_id": "q", "question": "cost?", "answer": [10], "mode": "all",
         "retrieval_correct": 1, "answer_correct": 1, "pred": "10"}
    with pytest.raises(ValueError, match="question differs"):
        validate_answers({"q": r}, {"q": a}, "fixture")
