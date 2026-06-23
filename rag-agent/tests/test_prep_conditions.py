"""Unit tests for the preprocessing-condition serializers (no data needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.prep.conditions import PrepTable, serialize  # noqa: E402
from rag_agent.prep.stats import (  # noqa: E402
    mrr, paired_delta_bootstrap, recall_at_k,
)
from rag_agent.prep.synth import TemplateSynth  # noqa: E402


FLAT = PrepTable(
    table_id="t1",
    columns=["Player", "Team", "Goals"],
    rows=[["Messi", "Inter Miami", "25"], ["Son", "Tottenham", "17"]],
    page_title="2024 season",
    section_title="Top scorers",
    caption="League goals",
)

HIER = PrepTable(
    table_id="t2",
    columns=["2017", "2018"],
    rows=[["10", "20"], ["30", "40"]],
    page_title="budget report",
    col_paths=[["actual", "2017"], ["actual", "2018"]],
    row_paths=[["revenue", "domestic"], ["revenue", "foreign"]],
)


def test_c0_is_table_only():
    s = serialize(FLAT, "C0")
    assert "Messi" in s and "Player | Team | Goals" in s
    assert "2024 season" not in s and "Top scorers" not in s


def test_c1_adds_metadata():
    s = serialize(FLAT, "C1")
    assert "Title: 2024 season" in s
    assert "Section: Top scorers" in s
    assert "Caption: League goals" in s
    assert "Messi" in s


def test_c2_adds_schema_with_types():
    s = serialize(FLAT, "C2")
    assert "Title: 2024 season" in s          # cumulative over C1
    assert "Goals (number)" in s
    assert "Player (text)" in s
    assert "e.g. Messi" in s


def test_c2h_spells_out_header_paths():
    s = serialize(HIER, "C2h")
    assert "actual > 2017" in s
    assert "revenue > domestic" in s


def test_c3_includes_synthetic_questions():
    s = serialize(FLAT, "C3", synth_provider=TemplateSynth(3))
    assert "Questions answerable from this table:" in s
    assert "?" in s
    assert "Goals (number)" in s              # cumulative over C2


def test_c3_hier_uses_hier_schema():
    s = serialize(HIER, "C3", synth_provider=TemplateSynth(3))
    assert "actual > 2017" in s


def test_template_synth_deterministic():
    a = TemplateSynth(5)(FLAT)
    b = TemplateSynth(5)(FLAT)
    assert a == b and len(a) > 0
    assert all(q.endswith("?") for q in a)


def test_template_synth_hier_uses_paths():
    qs = TemplateSynth(5)(HIER)
    assert qs and any("revenue" in q for q in qs)


def test_max_rows_truncation():
    big = PrepTable(table_id="t3", columns=["a"],
                    rows=[[str(i)] for i in range(100)])
    s = serialize(big, "C0", max_rows=5)
    assert "4" in s and "99" not in s


def test_recall_and_mrr():
    ranks = [1, 3, None, 11]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.5
    assert abs(mrr(ranks) - (1 + 1 / 3 + 1 / 11) / 4) < 1e-9


def test_paired_bootstrap_detects_clear_gap():
    a = [1] * 80 + [None] * 20
    b = [None] * 50 + [1] * 50
    mean, lo, hi = paired_delta_bootstrap(a, b, k=1, n_iters=2000, seed=42)
    assert abs(mean - 0.3) < 1e-9
    assert lo > 0  # significant


def test_paired_bootstrap_null_includes_zero():
    a = [1, None] * 50
    b = [None, 1] * 50
    mean, lo, hi = paired_delta_bootstrap(a, b, k=1, n_iters=2000, seed=42)
    assert lo <= 0 <= hi


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
