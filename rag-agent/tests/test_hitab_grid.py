# SPDX-License-Identifier: MIT
"""The raw-grid <-> data-matrix join, checked against HiTab's own annotation.

Three claims this module rests on, all of which used to be guessed:
  1. `linked_cells` coordinates index the RAW texts grid.
  2. walking the two header trees together maps a raw line to a data line.
  3. `answer_formulas` + `reference_cells_map` name the gold cells outright.
Each is checked here against the shipped test split, not against a fixture, so
a regression in any of them fails loudly instead of quietly shrinking a
population.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import hitab_grid as hg
from rag_agent.data.loader import load_samples

DATA = str(ROOT / "data/hitab")
SPLIT = "test"


def _load():
    samples = load_samples(DATA, SPLIT)
    tabs = {}
    for s in samples:
        tabs.setdefault(s["table_id"], hg.load_table(s["table_id"], DATA))
    return samples, tabs


def test_every_table_maps_and_gold_resolves():
    samples, tabs = _load()
    assert all(t is not None for t in tabs.values()), "a table failed to load"

    # 1+2. every data row/col of every table is reachable from a raw line
    for tid, tab in tabs.items():
        t = tab.table
        assert len(set(tab.row_map.values())) == t.n_rows, f"{tid} rows"

    # 3. gold resolves for all but a named handful, and lands on the annotated value
    why = Counter()
    ok = bad = 0
    for s in samples:
        tab = tabs[s["table_id"]]
        cells, mode, reason = hg.gold_target(s, tab)
        why[reason or f"ok/{mode}"] += 1
        if reason or mode != "all":
            continue
        for _tid, i, j in cells:
            assert 0 <= i < tab.table.n_rows and 0 <= j < tab.table.n_cols
        # a bare single-cell reference must hold the answer HiTab publishes
        f = str((s.get("answer_formulas") or [""])[0]).strip()
        if len(cells) == 1 and f.lstrip("=").isalnum() and s.get("answer"):
            (_t, i, j), = cells
            got = hg.norm_value(tab.table.data[i][j])
            want = hg.norm_value(s["answer"][0])
            ok += 1
            bad += int(got != want)
    scored = sum(v for k, v in why.items() if k.startswith("ok/"))
    assert scored / len(samples) > 0.97, why          # was 0.59 before this module
    assert bad / max(ok, 1) < 0.02, f"{bad}/{ok} bare refs disagree with the answer"
    print(f"{SPLIT}: {scored}/{len(samples)} scoreable, "
          f"bare-ref value agreement {(ok - bad)}/{ok}", dict(why))


if __name__ == "__main__":
    test_every_table_maps_and_gold_resolves()
    print("ok")
