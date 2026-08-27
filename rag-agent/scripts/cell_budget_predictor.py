#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How many cells does THIS query need? Predict it from retrieval signal alone.

`results/adaptive_cells_ceiling_verdict.json` measured the ceiling: cutting the
cell arm at the rank of the last gold cell holds OSC exactly and moves EM +.0909
on AIT-QA (53:12, p=0) and +.0253 on HiTab. That oracle reads gold. This asks
what a policy can recover from signal the query already provides.

Target  k*(q) -- the 1-based rank of the LAST gold cell in the corpus ranking.
        Taking the top k* cells is the shortest complete prefix.
Features scale-free ones only, so a model fit on one corpus can be applied to
        another without refitting: score gaps as a FRACTION of the top score,
        the coefficient of variation of the top 50, how many distinct tables the
        top 10 and top 50 touch, and how much the two retrievers agree.

Under-predicting costs completeness, which the oracle never does, so the policy
is a prediction times a MARGIN, and the margin is chosen on train to hold a
target retention. The frontier this prints -- mean cells against retention -- is
what says whether the predictor is worth wiring into the pipeline at all.

    PYTHONPATH=.:scripts .venv/bin/python scripts/cell_budget_predictor.py \
        --fit hitab:train --eval hitab:dev --eval aitqa:all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.retrieve.hybrid_index import HybridIndex, _tokenize
from rag_agent.serialization.base import Chunk
import corpus_dump_vs_cell as M

# Dense-only on purpose. `agree20` (bm25/dense top-20 overlap) scored 0.095 in
# importance against cv50's 0.416, and dropping it moved the budget-restricted
# frontier by under a point -- while at RUN time it would force a full bm25 pass
# per query on top of the dense one the run already does.
FEATURES = ("gap12", "gap1_10", "gap1_50", "cv50", "tab10", "tab50", "qlen")


def build(spec: str, cell_scheme: str, embed_model: str, cache_dir: str,
          data_dir: str):
    ds, split = spec.split(":")
    if ds == "hitab":
        C = M.hitab_corpus(data_dir, split, f"hitab_{split}_lookup_all")
    elif ds == "aitqa":
        C = M.aitqa_corpus()
    elif ds == "realhitbench":
        C = M.realhitbench_corpus()
    else:
        raise SystemExit(f"unknown dataset {ds!r}")
    if cell_scheme in ("S3", "S3c", "mt2net"):
        from rag_agent.serialization.caption import caption_sentence
        from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,
                                                       STRUCTURAL_COMPACT)
        tmpl = {"mt2net": MT2NET, "S3c": STRUCTURAL_COMPACT}.get(cell_scheme, STRUCTURAL)
        C.cell_text[:] = [caption_sentence(C.title.get(t, ""), rp, cp, v, template=tmpl)
                          for (rp, cp, v), (t, _i, _j) in zip(C.cell_paths, C.cell_owner)]
    chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=x,
                    scheme=cell_scheme, kind="cell")
              for x, (t, i, j) in zip(C.cell_text, C.cell_owner)]
    enc = M._CachedEncoder(M.default_encoder(model_name=embed_model), cache_dir,
                           f"{ds}_{split}_{embed_model}_{cell_scheme}")
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    owner = np.array([{t: k for k, t in enumerate(C.tids)}[t]
                      for t, _i, _j in C.cell_owner])
    pos_of = {c: n for n, c in enumerate(C.cell_owner)}
    return C, ix, owner, pos_of


def rows_for(spec, C, ix, owner, pos_of, limit: int = 0, seed: int = 0):
    """Features for each query. ``limit`` subsamples, deterministically.

    The bm25 half of ``agree20`` scores the WHOLE corpus per query, and HiTab
    train is 291k cells -- hours for a fit that eight features on a few hundred
    rows already saturates. Subsampling the FIT set is the cheap half of that
    trade; every eval set is scored in full.
    """
    import random
    qs = list(C.queries)
    if limit and limit < len(qs):
        qs = random.Random(seed).sample(qs, limit)
    out = []
    for q in qs:
        sc = ix._dense_scores(q["question"])
        order = np.argsort(-sc)
        s = sc[order]
        pos = np.empty(len(order), dtype=np.int64)
        pos[order] = np.arange(len(order))
        gold = [pos_of[g] for g in map(tuple, q["gold_cells"]) if g in pos_of]
        if not gold:
            continue
        kstar = int(max(int(pos[g]) for g in gold)) + 1
        s1 = float(s[0]) or 1e-9
        out.append({"spec": spec, "query_id": q["query_id"], "kstar": kstar,
                    "gap12": float(s[0] - s[1]) / s1,
                    "gap1_10": float(s[0] - s[min(9, len(s) - 1)]) / s1,
                    "gap1_50": float(s[0] - s[min(49, len(s) - 1)]) / s1,
                    "cv50": float(np.std(s[:50]) / (abs(np.mean(s[:50])) + 1e-9)),
                    "tab10": int(len(set(owner[order[:10]].tolist()))),
                    "tab50": int(len(set(owner[order[:50]].tolist()))),
                    "qlen": len(_tokenize(q["question"]))})
    return out


def frontier(rows, pred, margins):
    k = np.array([r["kstar"] for r in rows], dtype=float)
    print(f"    {'margin':>7}{'retention':>11}{'mean cells':>12}{'median':>8}"
          f"   (oracle mean {k.mean():.1f}, median {np.median(k):.0f})")
    out = []
    for m in margins:
        khat = np.maximum(1, np.ceil(pred * m))
        keep = float((khat >= k).mean())
        out.append({"margin": m, "retention": round(keep, 4),
                    "mean_cells": round(float(khat.mean()), 1),
                    "median_cells": int(np.median(khat))})
        print(f"    {m:7.1f}{keep:11.1%}{khat.mean():12.1f}{np.median(khat):8.0f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="hitab:train")
    ap.add_argument("--eval", action="append", default=[])
    ap.add_argument("--cell-scheme", default="S2")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--save-model", default="",
                    help="pickle the fitted model plus the margin chosen on TRAIN")
    ap.add_argument("--target-retention", type=float, default=0.95,
                    help="margin is the smallest one holding this much completeness "
                         "on the FIT split, among queries a budget could complete")
    ap.add_argument("--fit-cells", type=float, default=32.0,
                    help="cells a budget holds, for deciding which fit queries are "
                         "servable at all (512 tokens / ~15.9 tok per S2 cell)")
    ap.add_argument("--max-fit", type=int, default=800,
                    help="subsample the FIT population to this many queries")
    ap.add_argument("--out", default="results/cell_budget_predictor.json")
    args = ap.parse_args()

    from sklearn.ensemble import GradientBoostingRegressor

    def load(spec, limit=0):
        C, ix, owner, pos_of = build(spec, args.cell_scheme, args.embed_model,
                                     args.cache_dir, args.data_dir)
        r = rows_for(spec, C, ix, owner, pos_of, limit=limit)
        print(f"[{spec}] {len(r)} queries, k* mean "
              f"{np.mean([x['kstar'] for x in r]):.1f}", flush=True)
        return r

    tr = load(args.fit, limit=args.max_fit)
    X = np.array([[r[f] for f in FEATURES] for r in tr])
    y = np.log1p([r["kstar"] for r in tr])
    model = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=200)
    model.fit(X, y)
    MARGINS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    out = {"fit_on": args.fit, "cell_scheme": args.cell_scheme,
           "features": list(FEATURES), "n_train": len(tr), "eval": {},
           # k* is heavy-tailed (HiTab dev: mean 419, MEDIAN 2), so a frontier
           # over all queries is dominated by a tail the budget cannot serve
           # anyway. Keep the per-query rows so the frontier can be recomputed
           # restricted to the queries a budget could actually complete.
           "rows": {}}
    print(f"\n  [fit {args.fit}]")
    out["eval"][args.fit] = frontier(tr, np.expm1(model.predict(X)), MARGINS)
    out["rows"][args.fit] = [{**r, "pred": float(v)}
                             for r, v in zip(tr, np.expm1(model.predict(X)))]
    for spec in args.eval:
        rows = load(spec)
        Xe = np.array([[r[f] for f in FEATURES] for r in rows])
        print(f"\n  [eval {spec}]")
        out["eval"][spec] = frontier(rows, np.expm1(model.predict(Xe)), MARGINS)
        out["rows"][spec] = [{**r, "pred": float(v)}
                             for r, v in zip(rows, np.expm1(model.predict(Xe)))]
    if args.save_model:
        # k* is heavy-tailed, so the margin is chosen only over the queries a
        # budget could complete -- the rest are OSC=0 with or without a cut.
        import pickle, sklearn
        k = np.array([r["kstar"] for r in tr], float)
        pr = np.expm1(model.predict(X))
        sv = k <= args.fit_cells
        chosen = None
        for m in [x / 10 for x in range(10, 201)]:
            kh = np.minimum(np.maximum(1, np.ceil(pr[sv] * m)), args.fit_cells)
            if float((kh >= k[sv]).mean()) >= args.target_retention:
                chosen = m
                break
        chosen = chosen or 20.0
        Path(args.save_model).write_bytes(pickle.dumps(
            {"model": model, "features": list(FEATURES), "margin": chosen,
             "fit_on": args.fit, "n_train": len(tr),
             "target_retention": args.target_retention,
             "sklearn": sklearn.__version__}))
        out["margin_chosen_on_fit"] = chosen
        out["saved_model"] = args.save_model
        print(f"\n  margin chosen on {args.fit}: {chosen} "
              f"(target retention {args.target_retention:.0%}, "
              f"servable {int(sv.sum())}/{len(tr)})")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
