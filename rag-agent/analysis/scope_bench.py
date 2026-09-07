#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""헤더 범위 색인 — 셀 문장 + 범위 문장, 넘긴 셀 수로 맞춘 all-covered (PREREG-2026-09-07-scope-index.md).

단위: cell / row(행 전체) / col(열 전체) / rgrp(행 조상 × 열) / cgrp(열 조상 × 행). 같은 셀 집합은 하나만 둔다.

  bench:  PYTHONPATH=.:scripts:analysis .venv/bin/python analysis/scope_bench.py bench --split dev \
              --population hitab_dev_corpus_arith --gold-file results/audit2/hitab_dev_corpus_arith_gold.json \
              --embed-model models/bge-base-cell-ft-p1 --out results/scope/p1_dev_arith.json
  pairs:  ... scope_bench.py pairs --split train --population hitab_train_fit_alltypes --out results/scope/train_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                    # noqa: E402
import corpus_dump_vs_cell as cdv                                     # noqa: E402
from cell_rank_dump import cell_texts                                 # noqa: E402
from header_path_coverage import load_corpus                          # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax      # noqa: E402
from rag_agent.serialization.base import Chunk                        # noqa: E402
from rag_agent.serialization.caption import effective_titles          # noqa: E402

BUDGETS = (1, 5, 10, 20, 50, 200)
P = " > ".join


def build_units(C, title_mode="page"):
    """-> list of (kind, table_id, text, frozenset(cells)) for the whole corpus; cells first."""
    pt = json.load(open("results/tableconf/totto_page_titles.json")) if title_mode == "page" else None
    ti = effective_titles(C.tids, C.title, C.cell_owner, C.cell_paths, title_mode, page_titles=pt)
    units = [("cell", o[0], x, frozenset([o])) for x, o in zip(cell_texts(C, "S3c", title_mode), C.cell_owner)]
    by_t = defaultdict(list)
    for o, (rp, cp, v) in zip(C.cell_owner, C.cell_paths):
        by_t[o[0]].append((o, tuple(rp), tuple(cp), v))
    for t, cells in by_t.items():
        T = f"In the table '{ti[t]}', " if ti[t] else "In this table, "
        seen = {frozenset([c[0]]) for c in cells}
        rows, cols = defaultdict(list), defaultdict(list)
        for o, rp, cp, v in cells:
            rows[o[1]].append((o, rp, cp, v)); cols[o[2]].append((o, rp, cp, v))

        def add(kind, text, cs):
            fs = frozenset(cs)
            if len(fs) >= 2 and fs not in seen:
                seen.add(fs); units.append((kind, t, text, fs))
        for i, cs in rows.items():
            rp = cs[0][1]
            add("row", f"{T}among {P(rp)}, the values are: " + "; ".join(f"{P(cp)} = {v}" for _o, _r, cp, v in cs),
                [c[0] for c in cs])
        for j, cs in cols.items():
            cp = cs[0][2]
            add("col", f"{T}for {P(cp)}, the values are: " + "; ".join(f"{P(rp)} = {v}" for _o, rp, _c, v in cs),
                [c[0] for c in cs])
        # ancestor groups: row-path prefix x one column, and col-path prefix x one row
        for j, cs in cols.items():
            cp = cs[0][2]
            for depth in range(1, max(len(c[1]) for c in cs)):
                groups = defaultdict(list)
                for o, rp, _c, v in cs:
                    if len(rp) > depth:
                        groups[rp[:depth]].append((o, rp, v))
                for pre, g in groups.items():
                    add("rgrp", f"{T}among {P(pre)}, the values of {P(cp)} are: " + "; ".join(f"{P(rp[depth:])} = {v}" for _o, rp, v in g),
                        [x[0] for x in g])
        for i, cs in rows.items():
            rp = cs[0][1]
            for depth in range(1, max(len(c[2]) for c in cs)):
                groups = defaultdict(list)
                for o, _r, cp, v in cs:
                    if len(cp) > depth:
                        groups[cp[:depth]].append((o, cp, v))
                for pre, g in groups.items():
                    add("cgrp", f"{T}among {P(rp)}, the values of {P(pre)} are: " + "; ".join(f"{P(cp[depth:])} = {v}" for _o, cp, v in g),
                        [x[0] for x in g])
    return units


def min_cover(units_of_table, gold):
    """smallest unit (by cell count) whose cells ⊇ gold, or None."""
    best, best_len = None, None
    for k, (kind, _t, _x, cs) in units_of_table:
        if gold <= cs and (best is None or len(cs) < best_len):
            best, best_len = k, len(cs)
    return best


def scores(ix, question, alpha):
    bm = ix._bm25_scores(question)
    if alpha == 0.0:
        return bm
    dn = ix._dense_scores(question)
    return dn if alpha == 1.0 else alpha * _minmax(dn) + (1 - alpha) * _minmax(bm)


def covered_budget(order, units, gold):
    """cells handed over at the moment gold is fully covered (1e9 if never within 5000 units)."""
    need, used = set(gold), 0
    for r, p in enumerate(order[:5000]):
        used += len(units[p][3]); need -= units[p][3]
        if not need:
            return used
    return 10 ** 9


def enc(model, cache_dir, tag):
    return cdv._CachedEncoder(default_encoder(model_name=model), cache_dir, tag)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bench", "pairs"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", required=True)
    ap.add_argument("--gold-file", default="")
    ap.add_argument("--embed-model", default="models/bge-base-cell-ft-p1")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--title-mode", default="page")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--neg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.dataset, a.data_dir, a.rhb_question_types, a.rhb_em_only = "hitab", "data/hitab", [], False
    C = load_corpus(a)
    pop = C.queries
    if a.gold_file:
        G = json.load(open(a.gold_file))
        pop = [q | {"gold_cells": {tuple(x) for x in G[q["query_id"]]}} for q in pop if q["query_id"] in G]
    t0 = time.time()
    units = build_units(C, a.title_mode)
    kinds = defaultdict(int)
    for u in units:
        kinds[u[0]] += 1
    by_table = defaultdict(list)
    for k, u in enumerate(units):
        by_table[u[1]].append((k, u))
    print(f"[units] {dict(kinds)} in {time.time() - t0:.0f}s | pop {len(pop)}", flush=True)

    if a.mode == "pairs":
        rng = random.Random(a.seed)
        rows, stat = [], defaultdict(int)
        for q in pop:
            gold = set(q["gold_cells"]); t = q["gold_table"]
            k = min_cover(by_table[t], gold)
            targets = [units[k]] if k is not None else [u for _k, u in by_table[t] if u[0] == "cell" and next(iter(u[3])) in gold]
            for pos in targets:
                same = [u for _k, u in by_table[t] if u[0] == pos[0] and not (u[3] & gold)]
                negs = rng.sample(same, min(a.neg, len(same)))
                if len(negs) < a.neg:
                    continue
                stat[pos[0]] += 1
                rows.append({"anchor": q["question"], "positive": pos[2],
                             **{f"negative_{i + 1}": n[2] for i, n in enumerate(negs)}})
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[pairs] {len(rows)} rows, positives by kind {dict(stat)} -> {a.out}")
        return 0

    E = enc(a.embed_model, a.cache_dir, f"hitab_{a.split}_{a.embed_model}")
    res = {"n": len(pop), "units": dict(kinds), "embed_model": a.embed_model, "alpha": a.alpha}
    arms = {"cell_only": [k for k, u in enumerate(units) if u[0] == "cell"],
            "mixed": list(range(len(units))),
            "scope_only": [k for k, u in enumerate(units) if u[0] != "cell"]}
    # table chunk arm for the B=200 reference: title + header labels, covers every cell
    tab_units = [("table", t, C.table_text[t], frozenset(o for o in C.cell_owner if o[0] == t)) for t in C.tids]
    for name, idx in list(arms.items()) + [("table", None)]:
        t0 = time.time()
        U = tab_units if name == "table" else [units[k] for k in idx]
        ix = HybridIndex([Chunk(table_id="", chunk_id=str(n), text=u[2], scheme=name, kind="x") for n, u in enumerate(U)],
                         encoder=E, alpha=0.5)
        used = [covered_budget(np.argsort(-scores(ix, q["question"], a.alpha)), U, q["gold_cells"]) for q in pop]
        n = len(used)
        res[name] = {f"@{b}cells": round(sum(u <= b for u in used) / n, 4) for b in BUDGETS}
        res[name]["median_cells_when_covered"] = int(np.median([u for u in used if u < 10 ** 9])) if any(u < 10 ** 9 for u in used) else None
        print(f"  {name:<10} " + " ".join(f"@{b}={res[name][f'@{b}cells']:.4f}" for b in BUDGETS) + f"  ({time.time() - t0:.0f}s)", flush=True)
        ix.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
