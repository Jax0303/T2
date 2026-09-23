# SPDX-License-Identifier: MIT
"""Strict Recall: 리더에게 넘긴 셀 집합 전체를 K 없이, 좌표로 채점한다."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval import strict_recall as sr                        # noqa: E402

A, B, C, D = (("T", 0, j) for j in range(4))
DATA = str(ROOT / "data/hitab")


def test_all_gold_exactly():
    s = sr.score({A, B, C}, {A, B, C})
    assert (s["strict_recall"], s["exact_set_match"], s["precision"], s["recall"]) == (1, 1, 1, 1)


def test_all_gold_plus_an_extra_cell():
    s = sr.score({A, B, C}, {A, B, C, D})
    assert (s["strict_recall"], s["exact_set_match"], s["precision"]) == (1, 0, 0.75)
    assert s["irrelevant_predicted_cells"] == [D] and s["fp"] == 1


def test_one_gold_missing():
    s = sr.score({A, B, C}, {A, B})
    assert s["strict_recall"] == 0 and s["recall"] == pytest.approx(2 / 3)
    assert s["missing_gold_cells"] == [C] and s["fn"] == 1


def test_empty_prediction_is_zero():
    s = sr.score({A, B, C}, set())
    assert (s["precision"], s["recall"], s["f1"], s["strict_recall"]) == (0, 0, 0, 0)


def test_duplicate_cells_count_once():
    r = sr.row("d", "test", "q", "multi_cell", [A, B, C], [A, A, B, B, C, C])
    assert r["num_predicted_cells"] == 3 and r["precision"] == 1 and r["exact_set_match"] == 1


def test_selector_output_is_scored_whole_not_recut():
    from scripts import retrieval_accuracy as ra
    covers = [frozenset(("T", 0, j) for j in range(25)), frozenset([("T", 1, 0)])]
    got = ra.budget_select(range(2), covers, ["row0", "row1"], budget=20, dump=0).cells
    r = sr.row("d", "test", "q", "single_cell", [("T", 1, 0)], got)
    assert r["num_predicted_cells"] == 25 and r["strict_recall"] == 0   # last unit kept whole


def test_threshold_selection_is_variable_length_and_never_calls_budget_select():
    import numpy as np
    from scripts import retrieval_accuracy as ra
    covers = [frozenset([("T", 0, j)]) for j in range(5)]
    texts = [f"c{j}" for j in range(5)]
    before = ra.budget_select.calls
    got = ra.threshold_select(np.array([.9, .2, .75, .1, 1.0], np.float32), np.arange(5),
                              covers, texts, 0.7, dump=1)
    assert got.cells == {("T", 0, 4), ("T", 0, 0), ("T", 0, 2)}
    assert got.context == ["c4", "c0", "c2"]                 # best first, length set by score
    empty = ra.threshold_select(np.array([.3, .2]), np.array([3, 1]), covers, texts, 0.7, dump=0)
    assert empty.cells == set() and empty.units == []       # nothing is added to fill it
    assert ra.budget_select.calls == before


CASES = [({A}, None, [(0.9, {A}), (0.5, {B}), (0.1, {C})]),
         ({B, C}, None, [(0.8, {B}), (0.7, {C}), (0.2, {A})])]


def test_threshold_is_chosen_by_macro_f1_never_returning_everything():
    rows = sr.sweep(CASES, (0.0, 0.3, 0.6, 0.85))
    assert [round(r["macro_f1"], 4) for r in rows] == [0.65, 0.8333, 1.0, 0.5]
    assert sr.choose_threshold(rows) == 0.6
    assert sr.choose_threshold(sr.sweep(CASES, (0.6, 0.65))) == 0.65     # tie -> higher
    with pytest.raises(ValueError):
        sr.choose_threshold(sr.sweep(CASES, (0.0,)))                      # returns every cell


def test_threshold_is_applied_only_from_another_split(tmp_path):
    cfg = {"unit": "cell", "alpha": 0.7}
    t = sr.selected_threshold(sr.sweep(CASES, (0.3, 0.6)), "dev", "score", cfg)
    path = tmp_path / "dev_summary.json"
    path.write_text(json.dumps({"dataset": "hitab", "split": "dev", "threshold": t}))
    got = sr.load_threshold(path, "hitab", "test", cfg)
    assert (got["selected_threshold"], got["applied_threshold"], got["selected_on_split"]) == (0.6, 0.6, "dev")
    with pytest.raises(ValueError):
        sr.load_threshold(path, "hitab", "dev", cfg)
    with pytest.raises(ValueError):
        sr.load_threshold(path, "hitab", "test", {"unit": "cell", "alpha": 1.0})
    with pytest.raises(ValueError):
        sr.selected_threshold(sr.sweep(CASES, (0.3,)), "test", "score", cfg)


def test_same_value_at_other_coordinates_is_not_gold():
    grid = {("T", 0, 0): "7", ("T", 1, 1): "7"}
    assert grid[("T", 0, 0)] == grid[("T", 1, 1)]
    assert sr.score({("T", 0, 0)}, {("T", 1, 1)})["strict_recall"] == 0


def test_header_gold_is_found_through_a_cell_that_carries_it():
    h = ("T", "header", 5, 0)
    carriers = {h: frozenset({A, B})}
    s = sr.score({h}, {B, ("T", 9, 9)}, carriers)
    assert (s["strict_recall"], s["precision"], s["exact_set_match"]) == (1, 0.5, 0)
    assert s["irrelevant_predicted_cells"] == [("T", 9, 9)]
    assert sr.score({h}, {("T", 9, 9)}, carriers)["strict_recall"] == 0


def test_summary_puts_precision_and_size_next_to_strict_recall(tmp_path):
    rows = [sr.row("d", "s", "q1", "multi_cell", {A, B, C}, {A, B, C, D}),
            sr.row("d", "s", "q2", "single_cell", {A}, set())]
    s = sr.summarize(rows)
    assert (s["N"], s["strict_recall"], s["exact_set_match"]) == (2, 0.5, 0)
    assert s["macro_precision"] == 0.375 and s["micro_precision"] == 0.75
    assert s["micro_recall"] == 0.75 and s["macro_recall"] == 0.5
    assert (s["mean_predicted_cells"], s["median_predicted_cells"], s["p95_predicted_cells"]) == (2, 2, 4)
    assert sr.summarize([]) == {"N": 0}
    groups = {"overall": lambda r: True, "arithmetic": lambda r: r["question_type"] == "arithmetic"}
    out = sr.write(tmp_path, "x_strict_no_k", rows, groups, {"dataset": "d"})
    assert out["groups"]["arithmetic"] == {"N": 0}
    assert all(p.exists() for p in sr.outputs(tmp_path, "x_strict_no_k"))
    assert "group,N,strict_recall" in sr.outputs(tmp_path, "x_strict_no_k")[2].read_text()
    with pytest.raises(FileExistsError):
        sr.write(tmp_path, "x_strict_no_k", rows, groups, {})


def _hitab(pred):
    if not Path(DATA, "data", "test_samples.jsonl").exists():
        pytest.skip("HiTab data not present")
    from rag_agent.data.loader import load_samples
    for s in load_samples(DATA, "test"):
        if pred(s):
            return s, hg.load_table(s["table_id"], DATA)
    raise AssertionError("no such sample")


def test_hitab_arithmetic_gold_is_every_answer_operand():
    s, tab = _hitab(lambda s: s["aggregation"] == ["diff"]
                    and len(s["linked_cells"]["quantity_link"].get("[ANSWER]", {})) >= 2)
    gold, carriers, source, why, anomaly = hg.answer_gold(s, tab)
    refs, _ = hg.formula_refs(s)
    assert (why, anomaly, carriers, source) == (None, None, {}, "[ANSWER]")
    assert len(gold) >= 2 and gold == {(tab.table_id, *tab.to_data(rc)) for rc in refs}


def test_hitab_header_answer_gold_is_the_header_cell_its_cells_carry():
    s, tab = _hitab(lambda s: any("[ANSWER]" in v for v in s["linked_cells"]["entity_link"].values()))
    gold, carriers, source, why, _ = hg.answer_gold(s, tab)
    assert why is None and source == "[ANSWER]"
    t = tab.table
    for h in gold:
        assert h[1] == "header" and carriers[h]
        label = hg.norm_value(tab.raw["texts"][h[2]][h[3]])
        for _tid, i, j in carriers[h]:          # the cell sentence renders this header
            assert label in {hg.norm_value(x) for x in (*t.row_path(i), *t.col_path(j))}


HTML = ("<table><tr><td></td><td>2019</td><td>2018</td></tr>"
        "<tr><td>Revenue</td><td>5</td><td>7</td></tr>"
        "<tr><td>Cost</td><td>3</td><td>5</td></tr></table>")


def test_mh_gold_spans_tables_with_dataset_cell_ids():
    import mh_arms as mh
    tables, hdr = mh.build_tables({"u": ([HTML, HTML], "{}", [])})
    live = {(tid, i, j) for tid, tab in tables.items()
            for i, row in enumerate(tab.table.data) for j, v in enumerate(row) if v.strip()}
    (q,) = mh.resolve_gold([{"uid": "u", "coords": [(0, 2, 2), (1, 2, 2)]}], tables, hdr, live)
    assert q["excluded"] is None
    gold = {mh.cell_id(c, hdr) for c in q["gold"]}
    assert gold == {"u::0-2-2", "u::1-2-2"}
    r = sr.row("multihiertt", "validation", "u", "multi_cell", gold, {"u::0-2-2"},
               table_of=lambda x: x.rsplit("-", 2)[0], table_metrics=True)
    assert r["gold_tables"] == ["u::0", "u::1"] and r["predicted_tables"] == ["u::0"]
    assert r["derived_table_coverage"]["strict_recall"] == 0
    assert r["derived_table_coverage"]["recall"] == 0.5
    # "5" sits at 0-1-1 and 0-2-2 of one table: equal values, different cells
    assert tables["u::0"].table.grid[1][1] == tables["u::0"].table.grid[2][2] == "5"
    assert sr.score({"u::0-2-2"}, {"u::0-1-1"})["strict_recall"] == 0


def test_mh_text_only_questions_are_excluded_hybrids_kept(monkeypatch):
    import datasets
    import mh_arms as mh
    base = {"question": "q", "answer": "1", "program": "", "tables": [HTML],
            "table_description": "{}", "paragraphs": []}
    rows = [dict(base, uid="t", table_evidence=["0-2-2"], text_evidence=[]),
            dict(base, uid="h", table_evidence=["0-2-2"], text_evidence=[3],
                 program="add(5, 3), divide(#0, const_2)"),
            dict(base, uid="x", table_evidence=[], text_evidence=[3])]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(mh, "official_answers", lambda split: {r["uid"]: r["answer"] for r in rows})
    qs, _docs, skipped = mh.load_population("validation", keep_hybrid=True)
    assert [(q["uid"], q["has_text_evidence"], q["kind"]) for q in qs] == [
        ("t", False, "lookup"), ("h", True, "arith")]
    assert qs[1]["program_ops"] == ["add", "divide"] and skipped == {"text_only": 1}
    qs, _docs, skipped = mh.load_population("validation")
    assert [q["uid"] for q in qs] == ["t"] and skipped == {"needs_text_evidence": 2}


def test_mh_anomaly_log_flags_a_value_that_disagrees():
    import mh_arms as mh
    tables, _hdr = mh.build_tables({"u": ([HTML], "{}", [])})
    desc = {"0-2-2": "Table 0 shows Cost of 2018 is 5 .", "0-2-1": "Table 0 shows Cost of 2019 is 9 ."}
    docs = {"u": ([HTML], json.dumps(desc), [])}
    q = {"uid": "u", "coords": [(0, 2, 2), (0, 2, 1), (0, 1, 2)]}
    bad = mh.evidence_anomalies(q, docs, tables)
    assert [(b["cell"], b["problem"]) for b in bad] == [
        ("0-2-1", "value_differs"), ("0-1-2", "no_table_description_sentence")]
