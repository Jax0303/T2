#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""What the retrieval pipeline used to index and score, and what it does now.

Every count in the 2026-09-08 report comes from here, not from a doc. The two
retired rules are still runnable, so the "before" column is measured rather than
remembered:

  old table gate   a table was indexed only if
                   ``point3_reconstruction_cost.build_table_paths`` returned
                   non-None -- i.e. only if the header-reconstruction
                   experiment's value-match alignment cleared 0.90.
  old gold rule    gold cells came from ``quantity_link`` entries whose value
                   parsed as a number; anything else resolved to no gold and the
                   query left the population.

  PYTHONPATH=.:scripts .venv/bin/python analysis/pipeline_audit.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from point3_reconstruction_cost import build_table_paths                  # noqa: E402
from rag_agent.bench import hitab_grid as hg                              # noqa: E402
from rag_agent.data.loader import load_samples                            # noqa: E402
from rag_agent.stores.original_store import _to_float                     # noqa: E402


def old_gold_coords(sample: dict) -> int:
    """How many gold coords the retired ``_coords_of`` rule would have found."""
    ql = (sample.get("linked_cells") or {}).get("quantity_link") or {}
    n = 0
    for bucket in ql.values():
        if isinstance(bucket, dict):
            n += sum(1 for k, v in bucket.items()
                     if _to_float(v) is not None and hg.parse_coord(k))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--out", default="results/retrieval_accuracy/PIPELINE_AUDIT.json")
    a = ap.parse_args()

    samples = load_samples(a.data_dir, a.split)
    tids = sorted({s["table_id"] for s in samples})
    tabs = {t: hg.load_table(t, a.data_dir) for t in tids}

    raw_dir = Path(a.data_dir) / "data/tables/raw"
    old_ok = 0
    for tid in tids:
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.loads(f.read_text())
        except Exception:
            continue
        bt = tabs[tid]
        if bt is not None and build_table_paths(raw, bt.table) is not None:
            old_ok += 1

    cells_now = sum(sum(1 for r in t.table.data for v in r if str(v).strip())
                    for t in tabs.values() if t)
    why = Counter()
    old_scored = 0
    for s in samples:
        tab = tabs[s["table_id"]]
        _cells, mode, reason = hg.gold_target(s, tab)
        why[reason or f"scored/{mode}"] += 1
        # the old population: table cleared the gate AND numeric quantity_link
        if old_gold_coords(s) and tab is not None:
            f = raw_dir / f"{s['table_id']}.json"
            if f.exists() and build_table_paths(json.loads(f.read_text()),
                                                tab.table) is not None:
                old_scored += 1

    out = {
        "split": a.split,
        "tables_referenced_by_split": len(tids),
        "tables_in_store": len(hg.table_ids(a.data_dir)),
        "tables_indexed_before": old_ok,
        "tables_indexed_now": sum(1 for t in tabs.values() if t is not None),
        "cells_indexed_now": cells_now,
        "queries_in_split": len(samples),
        "queries_scored_before": old_scored,
        "queries_scored_now": sum(v for k, v in why.items() if k.startswith("scored/")),
        "queries_scored_now_by_mode": {k.split("/")[1]: v for k, v in why.items()
                                       if k.startswith("scored/")},
        "queries_excluded_now": {k: v for k, v in why.items()
                                 if not k.startswith("scored/")},
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
