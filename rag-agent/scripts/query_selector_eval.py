#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Step 3 — does cutting the selector clause off the query find the operands?

The within-document ladder's largest failure bucket is "right table, wrong row":
the query names a selector ("in the year with the most Net revenues") next to
its target ("the growth rate of Pretax income"), and the retriever brings back
the selector's cells, which are never gold.

`rag_agent.query.selector.strip_selector` cuts the clause lexically. This runs
the ladder's S3 arm twice over the SAME corpus, gold and pool -- once with the
full question, once with the stripped one -- and reports both the whole
population and the subset the rule actually fires on, because a rule that fires
on a quarter of the queries can only move the overall number by a quarter of
its effect.

Run: PYTHONPATH=. .venv/bin/python scripts/query_selector_eval.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from operand_collision_multihiertt import (RETRIEVERS, _minmax, _tokenize,
                                           build_corpus, cell_text,
                                           load_population)
from rag_agent.query.selector import has_selector, strip_selector
from rag_agent.retrieve.encoders import default_encoder, default_prefixes
from rag_agent.runenv import run_env


def summarize(ranks):
    out = {"n_queries": len(ranks),
           "mrr": round(float(np.mean([1.0 / min(r) for r in ranks])), 4)}
    for k in (1, 5, 10, 20):
        out[f"hit@{k}"] = round(float(np.mean([min(r) <= k for r in ranks])), 4)
        out[f"set_em@{k}"] = round(float(np.mean([max(r) <= k for r in ranks])), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--scheme", default="S3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/query_selector_eval.json")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)

    from rank_bm25 import BM25Okapi

    queries, docs = load_population(args.max_queries, "arith_multi")
    tables, cells, pop = build_corpus(queries, docs)
    pool_of = {}
    for gi, c in enumerate(cells):
        pool_of.setdefault(c["table"][0], []).append(gi)

    full = [q["question"] for q in pop]
    stripped = [strip_selector(t) for t in full]
    fires = [i for i, t in enumerate(full) if stripped[i] != t]
    print(f"[pop] {len(pop)} queries | selector clause cut on {len(fires)} "
          f"({len(fires)/len(pop):.1%})", flush=True)
    for i in fires[:3]:
        print(f"  - {full[i][:90]}\n    -> {stripped[i][:90]}", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    qpre, ppre = default_prefixes(enc.name)
    texts = [cell_text(c, args.scheme) for c in cells]
    vecs = np.asarray(enc.encode([ppre + t for t in texts]))
    bm25_of = {uid: BM25Okapi([_tokenize(texts[gi]) for gi in idxs])
               for uid, idxs in pool_of.items()}

    results = {}
    per_arm_ranks = {}
    for arm, qtexts in (("full", full), ("stripped", stripped)):
        qv = np.asarray(enc.encode([qpre + t for t in qtexts]))
        by_ret = {name: [] for name, _ in RETRIEVERS}
        for qi, q in enumerate(pop):
            idxs = pool_of[q["uid"]]
            gold = {int(g) for g in q["gold"]}
            dn = _minmax(np.asarray([vecs[gi] for gi in idxs]) @ qv[qi])
            bm = _minmax(np.asarray(
                bm25_of[q["uid"]].get_scores(_tokenize(qtexts[qi])), dtype=np.float32))
            for name, alpha in RETRIEVERS:
                order = np.argsort(-(alpha * dn + (1.0 - alpha) * bm))
                rank_of = {}
                for pos, local in enumerate(order, 1):
                    gi = idxs[int(local)]
                    if gi in gold:
                        rank_of[gi] = pos
                        if len(rank_of) == len(gold):
                            break
                by_ret[name].append([rank_of[g] for g in gold])
        per_arm_ranks[arm] = by_ret
        results[arm] = {
            "all": {n: summarize(r) for n, r in by_ret.items()},
            "selector_subset": {n: summarize([r[i] for i in fires])
                                for n, r in by_ret.items()},
        }
        for n in by_ret:
            a = results[arm]["all"][n]
            s = results[arm]["selector_subset"][n]
            print(f"[{arm:8} {n:6}] all mrr={a['mrr']:.4f} h@1={a['hit@1']:.4f} | "
                  f"subset mrr={s['mrr']:.4f} h@1={s['hit@1']:.4f}", flush=True)

    # paired test on the queries the rule actually touched
    from scipy.stats import wilcoxon
    contrasts = {}
    for n in per_arm_ranks["full"]:
        a = [min(per_arm_ranks["full"][n][i]) for i in fires]
        b = [min(per_arm_ranks["stripped"][n][i]) for i in fires]
        d = [x - y for x, y in zip(a, b)]
        contrasts[n] = {
            "n_pairs": len(a), "n_changed": sum(1 for x in d if x),
            "improved": sum(1 for x in d if x > 0),
            "worsened": sum(1 for x in d if x < 0),
            "median_rank_full": statistics.median(a),
            "median_rank_stripped": statistics.median(b),
            "p_two_sided": round(float(wilcoxon(a, b).pvalue), 5) if any(d) else 1.0,
        }
    print("\n[paired, selector subset]", json.dumps(contrasts, indent=1))

    out = {"env": env,
           "population": {"name": "multihiertt_arith_multi_within_doc",
                          "n_queries": len(pop), "n_selector_queries": len(fires)},
           "scheme": args.scheme, "encoder": enc.name,
           "examples": [{"full": full[i], "stripped": stripped[i]} for i in fires[:15]],
           "by_arm": results,
           "paired_on_selector_subset": contrasts}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
