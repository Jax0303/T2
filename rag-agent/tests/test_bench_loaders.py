# SPDX-License-Identifier: MIT
"""Data-free unit tests for WikiSQL SQL execution and FinQA evidence parsing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.bench import wikisql, finqa


def test_wikisql_execute_select_eq():
    table = {
        "id": "t", "header": ["Name", "Pos", "Pts"],
        "rows": [["Ann", "G", "10"], ["Bob", "F", "20"], ["Cy", "G", "30"]],
    }
    sql = {"sel": 1, "agg": 0, "conds": {"column_index": [0], "operator_index": [0], "condition": ["Bob"]}}
    answer, matched = wikisql._execute(table, sql)
    assert answer == ["F"] and matched == [1]
    ops = wikisql._operands(table, sql, matched)
    # selected col (1) + condition col (0), in the matched row
    assert {(o.row, o.col) for o in ops} == {(1, 1), (1, 0)}


def test_wikisql_execute_sum_agg():
    table = {"id": "t", "header": ["Name", "Pos", "Pts"],
             "rows": [["Ann", "G", "10"], ["Bob", "F", "20"], ["Cy", "G", "30"]]}
    sql = {"sel": 2, "agg": 4, "conds": {"column_index": [1], "operator_index": [0], "condition": ["G"]}}
    answer, matched = wikisql._execute(table, sql)
    assert answer == [40.0] and matched == [0, 2]


def test_finqa_evidence_parsing():
    table = [
        ["company", "payments volume ( billions )", "cards ( millions )"],
        ["visa inc.", "$ 2457", "1592"],
        ["american express", "637", "86"],
    ]
    evidence = ["the american express of payments volume ( billions ) is 637 ; "
                "the american express of cards ( millions ) is 86 ;"]
    bt = finqa._bench_table(table, "t")
    ops = finqa._resolve_operands(table, bt, evidence)
    # american express is data row index 1 (header stripped); two columns referenced
    assert {(o.row, o.col) for o in ops} == {(1, 1), (1, 2)}
    assert any(o.value == 637.0 for o in ops)
    # header_path is the full path (row label > col header), matching candidate_paths
    assert ops[0].header_path == bt.full_path(ops[0].row, ops[0].col)
