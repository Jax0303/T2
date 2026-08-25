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
    aitqa_corpus, hitab_corpus, label_doc_freq, multihiertt_corpus,
    realhitbench_corpus, s2h_prefixes, table_top_labels,
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



def addresses(C, scheme: str = "S2", k: int = 1) -> list:
    """셀별 주소 문자열(값 제외). 사전등록이 인용하는 충돌률이 이 주소 기준이다."""
    base = [" > ".join(rp) + " | " + " > ".join(cp) for rp, cp, _v in C.cell_paths]
    if scheme == "S2":
        return base
    if scheme == "S2h_all":
        # 사전등록 §3의 문자 그대로: 겹치지 않는 최상위 축 레이블 전부. 충돌은 가장
        # 많이 줄지만 토큰이 3.6배가 되어 T5(+25%)를 깬다. 대조로만 남긴다.
        tops = table_top_labels(C)
        return [f"[{' | '.join(sorted(tops[t] - set(rp) - set(cp)))}] {a}"
                if (tops[t] - set(rp) - set(cp)) else a
                for (rp, cp, _v), (t, _i, _j), a
                in zip(C.cell_paths, C.cell_owner, base)]
    return [pre + a for pre, a in zip(s2h_prefixes(C, k), base)]

def measure(C) -> dict:
    n = len(C.cell_text)
    same, cross = _collide(C.cell_text, C.cell_owner)
    addr = addresses(C, "S2")
    a_same, a_cross = _collide(addr, C.cell_owner)
    schemes = {}
    for sch in ("S2h", "S2h_all"):
        alt = addresses(C, sch)
        s_same, s_cross = _collide(alt, C.cell_owner)
        changed = sum(1 for a, b in zip(addr, alt) if a != b)
        tok = lambda L: sum(len(x.split()) for x in L) / len(L)
        schemes[sch] = {
            "changed_frac": round(changed / n, 4),
            "addr_collision_frac": round((s_same + s_cross) / n, 4),
            "addr_collision_same_table": round(s_same / n, 4),
            "addr_collision_cross_table": round(s_cross / n, 4),
            "tokens_per_cell": round(tok(alt), 2),
            "tokens_per_cell_ratio": round(tok(alt) / tok(addr), 4),
        }
    rd = np.array([len(rp) for rp, _cp, _v in C.cell_paths])
    cd = np.array([len(cp) for _rp, cp, _v in C.cell_paths])
    titled = sum(1 for t in C.tids if C.title.get(t))
    return {
        "tables": len(C.tids), "cells": n, "queries": len(C.queries),
        "titled": titled, "titled_frac": round(titled / len(C.tids), 4),
        "collision_frac": round((same + cross) / n, 4),
        "collision_same_table": round(same / n, 4),
        "collision_cross_table": round(cross / n, 4),
        "addr_collision_frac": round((a_same + a_cross) / n, 4),
        "addr_collision_same_table": round(a_same / n, 4),
        "addr_collision_cross_table": round(a_cross / n, 4),
        "row_depth_mean": round(float(rd.mean()), 3),
        "col_depth_mean": round(float(cd.mean()), 3),
        "path_depth_mean": round(float((rd + cd).mean()), 3),
        "s2h": schemes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpora", default="",
                    help="쉼표로 고른 코퍼스만 잰다. 비우면 전부 -- 다만 hitab 외에는 "
                         "data/ 가 gitignore라 그 파일이 있는 기계에서만 돈다")
    ap.add_argument("--out", default="results/corpus_discriminability.json")
    args = ap.parse_args()

    corpora = {
        "hitab": lambda: hitab_corpus(args.data_dir, args.split, args.population),
        "aitqa": aitqa_corpus,
        "realhitbench": realhitbench_corpus,
        "multihiertt": lambda: multihiertt_corpus(args.mh_queries, args.seed),
    }
    if args.corpora:
        want = {x.strip() for x in args.corpora.split(",")}
        corpora = {k: v for k, v in corpora.items() if k in want}
    out = {"metric": "S2 cell-sentence uniqueness and title coverage, no reader",
           "corpora": {}}
    hdr = (f"{'corpus':14}{'tables':>7}{'cells':>8}{'titled':>9}"
           f"{'collide':>9}{'in-table':>10}{'x-table':>9}{'depth':>7}"
           f"{'addr':>8}{'addr-x':>8}")
    print(hdr)
    for name, build in corpora.items():
        m = measure(build())
        out["corpora"][name] = m
        print(f"{name:14}{m['tables']:7}{m['cells']:8}{m['titled_frac']:9.1%}"
              f"{m['collision_frac']:9.1%}{m['collision_same_table']:10.1%}"
              f"{m['collision_cross_table']:9.1%}{m['path_depth_mean']:7.2f}"
              f"{m['addr_collision_frac']:8.1%}{m['addr_collision_cross_table']:8.1%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
