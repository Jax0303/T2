#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""McNemar significance check on the differences reported so far in
results/embedding_fusion_diagnosis_20260917/dense_representation_sweep.json --
none of those R@1 deltas among n=1..4 were tested for significance, only
printed as point estimates. This asks: of the ~1-30 query flips behind each
delta, is that more than coin-flip noise at n=991?

Pairs tested (all same 991-query population, paired per-query correct/wrong):
  plain            vs leaf_both_x1 (dense-only, sparse=plain) -- does leaf-repeat
                   help AT ALL, beyond noise?
  leaf_both_x1     vs leaf_both_x2 (dense-only)                -- the claimed
                   "peak" step, +.0051
  leaf_both_x2     vs structural_leaf (deployed, both channels x1) -- the
                   claimed "new best beats deployed", +.0010

  PYTHONPATH=. .venv/bin/python scripts/leaf_significance_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402
from scipy.stats import binomtest                                     # noqa: E402

from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from bottleneck_diagnosis import load_primary_population               # noqa: E402
from sentence_disambiguation_eval import encode_corpus                # noqa: E402
from dense_representation_sweep import ALPHA, build_corpus            # noqa: E402

DISAMB_DIR = ROOT / "results/sentence_disambiguation_20260916"


def hits_for(name: str, emb: dict, bm_plain, coord_index, ordered, qvecs, sparse_by_q) -> dict:
    out = {}
    for qid, r in ordered:
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            continue
        dense = emb[name] @ qvecs[qid]
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse_by_q[qid])
        out[qid] = bool(np.argmax(hybrid) == gidx)
    return out


def hits_from_jsonl(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "excluded" in r:
                continue
            out[r["query_id"]] = bool(r["esm"])
    return out


def mcnemar(a: dict, b: dict, label: str) -> None:
    qids = sorted(set(a) & set(b))
    a_only = sum(a[q] and not b[q] for q in qids)
    b_only = sum(b[q] and not a[q] for q in qids)
    both = sum(a[q] and b[q] for q in qids)
    neither = sum(not a[q] and not b[q] for q in qids)
    n_disc = a_only + b_only
    p = binomtest(min(a_only, b_only), n_disc, 0.5).pvalue if n_disc else 1.0
    print(f"{label}: n={len(qids)} a_acc={sum(a.values())/len(a):.4f} "
         f"b_acc={sum(b.values())/len(b):.4f} a_only={a_only} b_only={b_only} "
         f"both={both} neither={neither} discordant={n_disc} p={p:.4f}")


def main() -> int:
    pop = load_primary_population()
    ordered = sorted(pop.items())
    names_needed = ("plain", "leaf_both_x1", "leaf_both_x2")
    texts, coords = build_corpus()
    coord_index = {c: idx for idx, c in enumerate(coords)}

    enc = default_encoder()
    emb = {n: encode_corpus(texts[n], enc)[0] for n in names_needed}
    bm_plain = SparseBM25(_tokenize(t) for t in texts["plain"])
    qvecs = {qid: enc.encode_query([r["question"]])[0].astype(np.float32) for qid, r in ordered}
    sparse_by_q = {qid: bm_plain.get_scores(_tokenize(r["question"])) for qid, r in ordered}

    hits = {n: hits_for(n, emb, bm_plain, coord_index, ordered, qvecs, sparse_by_q)
           for n in names_needed}
    hits["structural_leaf_deployed"] = hits_from_jsonl(DISAMB_DIR / "retrieval_structural_leaf.jsonl")

    mcnemar(hits["plain"], hits["leaf_both_x1"], "plain vs leaf_both_x1(dense-only)")
    mcnemar(hits["leaf_both_x1"], hits["leaf_both_x2"], "leaf_both_x1 vs leaf_both_x2(dense-only)")
    mcnemar(hits["leaf_both_x2"], hits["structural_leaf_deployed"],
           "leaf_both_x2(dense-only) vs structural_leaf(deployed, both-channel x1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
