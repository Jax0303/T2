#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How distinguishable are a corpus's cell sentences, before any reader runs?

The thesis is that a cell's HEADER PATH, written as a sentence, tells similar
cells apart well enough to retrieve the right one. That claim has a metric of
its own that needs no LLM and no budget: how often do two different cells
produce the SAME sentence? Where they collide, no encoder can separate them --
the ceiling is set by the serialization, not by the model.

Reported per corpus:

  titled          fraction of tables shipping a title/caption. The title is the
                  only TABLE-level term in an S3 sentence, and three of the four
                  corpora here do not have one, which is why the S3 gain does
                  not cross datasets.
  collision       fraction of cells whose S2 sentence is not unique, split into
                  collisions INSIDE one table and collisions ACROSS tables. The
                  split is the finding: within a table the header path is
                  already near-unique everywhere, so the discriminability that
                  is missing is between tables, not inside them.
  addr collision  the same, on the ADDRESS alone (paths, value stripped). This
                  is the figure PREREG-2026-08-25-unique-tag.md reports; the
                  full-sentence figure is lower because a distinct value can
                  separate two cells that share an address, which retrieval
                  cannot rely on -- the query holds the address, not the value.
  path depth      row-path + col-path length, i.e. how much hierarchy there is
                  to serialize in the first place -- the control for reading
                  the collision numbers as "these tables are just flatter".

Run:
    PYTHONPATH=.:scripts .venv/bin/python scripts/corpus_discriminability.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from corpus_dump_vs_cell import (
    aitqa_corpus, hitab_corpus, multihiertt_corpus, realhitbench_corpus,
    s2h_prefixes,
)


def _collide(keys, owner) -> tuple[int, int]:
    """(inside-one-table, across-tables) cell counts sharing a key."""
    owners = defaultdict(list)
    for k, (tid, _i, _j) in zip(keys, owner):
        owners[k].append(tid)
    same = cross = 0
    for tids in owners.values():
        if len(tids) < 2:
            continue
        # a key shared by two tables cannot be fixed by anything the header
        # path knows; one shared inside a table can
        if len(set(tids)) > 1:
            cross += len(tids)
        else:
            same += len(tids)
    return same, cross


def measure(C, scheme: str = "S2") -> dict:
    """The T0 numbers for one corpus under one cell scheme.

    ``scheme`` other than S2 prefixes every cell with the table-level
    discriminator of PREREG-2026-08-25-structural-discriminator.md §3. The
    prefix goes on the ADDRESS as well as the sentence: it is part of what the
    encoder sees, and the address (value stripped) is what a query can match.
    """
    n = len(C.cell_text)
    pre = [""] * n if scheme == "S2" else s2h_prefixes(C, scheme)
    text = [f"[{x}] {t}" if x else t for x, t in zip(pre, C.cell_text)]
    addr = [(f"[{x}] " if x else "") + " > ".join(rp) + " | " + " > ".join(cp)
            for x, (rp, cp, _v) in zip(pre, C.cell_paths)]
    same, cross = _collide(text, C.cell_owner)
    a_same, a_cross = _collide(addr, C.cell_owner)
    rd = np.array([len(rp) for rp, _cp, _v in C.cell_paths])
    cd = np.array([len(cp) for _rp, cp, _v in C.cell_paths])
    titled = sum(1 for t in C.tids if C.title.get(t))
    return {
        "scheme": scheme,
        "tables": len(C.tids), "cells": n, "queries": len(C.queries),
        "titled": titled, "titled_frac": round(titled / len(C.tids), 4),
        # T0: does the scheme actually change the sentence, and what does the
        # change cost? A scheme that is S2 on most cells cannot buy anything,
        # and one that doubles the cell blows T5's +25% token cap.
        "differs_frac": round(sum(1 for x in pre if x) / n, 4),
        "tokens_per_cell": round(float(np.mean([len(t.split()) for t in text])), 3),
        "collision_frac": round((same + cross) / n, 4),
        "collision_same_table": round(same / n, 4),
        "collision_cross_table": round(cross / n, 4),
        "addr_collision_frac": round((a_same + a_cross) / n, 4),
        "addr_collision_same_table": round(a_same / n, 4),
        "addr_collision_cross_table": round(a_cross / n, 4),
        "row_depth_mean": round(float(rd.mean()), 3),
        "col_depth_mean": round(float(cd.mean()), 3),
        "path_depth_mean": round(float((rd + cd).mean()), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell-scheme", default="S2",
                    help="comma-separated schemes to measure on the same corpus, "
                         "e.g. 'S2,S2h,S2hr'. S2 first is what the others are "
                         "read against (T0 of the structural-discriminator "
                         "prereg: does the scheme change the sentence at all, "
                         "does it cut cross-table collision, what does it cost)")
    ap.add_argument("--out", default="results/corpus_discriminability.json")
    args = ap.parse_args()
    schemes = [x.strip() for x in args.cell_scheme.split(",") if x.strip()]

    corpora = {
        "hitab": lambda: hitab_corpus(args.data_dir, args.split, args.population),
        "aitqa": aitqa_corpus,
        "realhitbench": realhitbench_corpus,
        "multihiertt": lambda: multihiertt_corpus(args.mh_queries, args.seed),
    }
    out = {"metric": "cell-sentence uniqueness and title coverage, no reader",
           "schemes": schemes, "corpora": {}}
    hdr = (f"{'corpus':14}{'scheme':7}{'tables':>7}{'cells':>8}{'titled':>9}"
           f"{'collide':>9}{'in-table':>10}{'x-table':>9}{'depth':>7}"
           f"{'addr':>8}{'addr-x':>8}{'differs':>9}{'tok/cell':>9}")
    print(hdr)
    for name, build in corpora.items():
        C = build()
        for sch in schemes:
            m = measure(C, sch)
            out["corpora"].setdefault(name, {})[sch] = m
            print(f"{name:14}{sch:7}{m['tables']:7}{m['cells']:8}{m['titled_frac']:9.1%}"
                  f"{m['collision_frac']:9.1%}{m['collision_same_table']:10.1%}"
                  f"{m['collision_cross_table']:9.1%}{m['path_depth_mean']:7.2f}"
                  f"{m['addr_collision_frac']:8.1%}{m['addr_collision_cross_table']:8.1%}"
                  f"{m['differs_frac']:9.1%}{m['tokens_per_cell']:9.2f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
