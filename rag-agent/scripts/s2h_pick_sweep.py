#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Which table-level discriminator fits under the token cap -- and where it cannot.

T0 of PREREG-2026-08-25-structural-discriminator.md asks two things of a scheme
before any EM is read: it must actually change the sentence, and it must cut
cross-table address collision at least in half. `corpus_discriminability.py
--cell-scheme` answers those for the two schemes the prereg names. This script
answers the question that opens once they are measured: **§3 as written blows the
+25% token cap of T5 by 8x, so is there a cheaper pick rule that still halves the
collision?**

Two parts, neither using an LLM or an encoder:

  pick rules   the prefix is one root (or one row root + one col root) chosen by
               table-frequency, optionally truncated to the first N words. Each
               is scored on cross-table ADDRESS collision and tokens per cell,
               against the S2 baseline of the same corpus.
  where it is  for the corpus that resists -- AIT-QA, the prereg's primary test
  unreachable  -- which TABLE PAIRS the surviving collisions run between. A term
               read off the header tree is identical for two tables with the same
               header tree, so a pair like that is unreachable by this family of
               schemes rather than by a particular pick rule.

    PYTHONPATH=.:scripts .venv/bin/python scripts/s2h_pick_sweep.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from corpus_discriminability import _collide
from corpus_dump_vs_cell import (aitqa_corpus, hitab_corpus, realhitbench_corpus)

CAP = 0.25          # T5: tokens per cell may grow by at most this much


def axis_roots(C) -> tuple[dict, dict]:
    """(row roots, col roots) per table -- the top level of each header forest."""
    rr, cr = defaultdict(dict), defaultdict(dict)
    for (rp, cp, _v), (t, _i, _j) in zip(C.cell_paths, C.cell_owner):
        if rp and str(rp[0]).strip():
            rr[t][str(rp[0]).strip()] = None
        if cp and str(cp[0]).strip():
            cr[t][str(cp[0]).strip()] = None
    return {t: list(v) for t, v in rr.items()}, {t: list(v) for t, v in cr.items()}


def _pick(roots, df, short_first: bool, trunc: int) -> str:
    if not roots:
        return ""
    key = ((lambda r: (df[r.lower()], len(r.split()), r)) if short_first
           else (lambda r: (df[r.lower()], r)))
    best = min(roots, key=key)
    return " ".join(best.split()[:trunc]) if trunc else best


def addresses(C, prefix: dict) -> list:
    """The retrievable string minus the value -- what a query has to match."""
    return [(f"[{prefix.get(t, '')}] " if prefix.get(t) else "")
            + " > ".join(rp) + " | " + " > ".join(cp)
            for (rp, cp, _v), (t, _i, _j) in zip(C.cell_paths, C.cell_owner)]


def score(C, prefix: dict) -> dict:
    n = len(C.cell_text)
    text = [f"[{prefix.get(t, '')}] {s}" if prefix.get(t) else s
            for s, (t, _i, _j) in zip(C.cell_text, C.cell_owner)]
    _same, cross = _collide(addresses(C, prefix), C.cell_owner)
    return {"addr_collision_cross_table": round(cross / n, 4),
            "tokens_per_cell": round(float(np.mean([len(t.split()) for t in text])), 3),
            "distinct_labels": len(set(prefix.values()))}


def unreachable_pairs(C) -> dict:
    """Which table pairs AIT-QA's cross-table collisions actually run between."""
    owners = defaultdict(list)
    for a, (t, _i, _j) in zip(addresses(C, {}), C.cell_owner):
        owners[a].append(t)
    pairs, cells = Counter(), 0
    for a, tids in owners.items():
        if len(set(tids)) < 2:
            continue
        cells += len(tids)
        for x in sorted(set(tids)):
            for y in sorted(set(tids)):
                if x < y:
                    pairs[(x, y)] += 1
    top = pairs.most_common(10)
    tot = sum(pairs.values())
    return {"cells_in_cross_table_collisions": cells,
            "cells_frac": round(cells / len(C.cell_text), 4),
            "table_pairs": len(pairs),
            "top10_share_of_colliding_addresses": round(sum(c for _p, c in top) / tot, 4),
            "top10": [{"a": x, "b": y, "shared_addresses": c} for (x, y), c in top]}


RULES = [("rare", False, 0), ("rare+trunc3", False, 3), ("rare+trunc2", False, 2),
         ("rare,short", True, 0), ("rare,short+trunc3", True, 3)]


def main() -> int:
    out = {"metric": "cross-table address collision and token cost per pick rule, "
                     "no reader and no encoder",
           "token_cap": CAP, "corpora": {}}
    for name, build in (("hitab", lambda: hitab_corpus("data/hitab", "dev",
                                                       "hitab_dev_lookup_all")),
                        ("aitqa", aitqa_corpus),
                        ("realhitbench", realhitbench_corpus)):
        C = build()
        n = len(C.cell_text)
        _same, cross = _collide(addresses(C, {}), C.cell_owner)
        base_x = cross / n
        base_tok = float(np.mean([len(t.split()) for t in C.cell_text]))
        rr, cr = axis_roots(C)
        dfr = Counter(r for rs in rr.values() for r in {x.lower() for x in rs})
        dfc = Counter(r for rs in cr.values() for r in {x.lower() for x in rs})
        rows = {"S2": {"addr_collision_cross_table": round(base_x, 4),
                       "tokens_per_cell": round(base_tok, 3),
                       "tables": len(C.tids)}}
        print(f"{name:13} {'S2':20} x-addr {base_x:6.1%}  tok {base_tok:6.2f}")
        for rn, short_first, trunc in RULES:
            for slots in ("one", "row+col"):
                pref = {t: (_pick(rr.get(t, []) + cr.get(t, []),
                                  dfr + dfc, short_first, trunc)
                            if slots == "one" else
                            " | ".join(x for x in
                                       (_pick(rr.get(t, []), dfr, short_first, trunc),
                                        _pick(cr.get(t, []), dfc, short_first, trunc))
                                       if x))
                        for t in C.tids}
                m = score(C, pref)
                grow = m["tokens_per_cell"] / base_tok - 1
                m["token_growth"] = round(grow, 4)
                m["halves_collision"] = m["addr_collision_cross_table"] <= base_x / 2
                m["under_cap"] = grow <= CAP
                rows[f"{rn}:{slots}"] = m
                print(f"{name:13} {rn + ':' + slots:20} "
                      f"x-addr {m['addr_collision_cross_table']:6.1%}  "
                      f"tok {m['tokens_per_cell']:6.2f} ({grow:+.1%})  "
                      f"labels {m['distinct_labels']:4}/{len(C.tids)}  "
                      f"halved={'Y' if m['halves_collision'] else 'N'} "
                      f"cap={'Y' if m['under_cap'] else 'N'}")
        out["corpora"][name] = rows
        if name == "aitqa":
            out["aitqa_collision_sources"] = unreachable_pairs(C)
    p = Path("results/s2h_pick_sweep.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
