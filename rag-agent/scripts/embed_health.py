"""Is the sentence embedding actually doing its job? Seven checks, one report.

Every retrieval number in this repo rests on one assumption that no result file
records: that the vectors under the index are the ones we think they are. A
hashed-bag-of-words fallback, an unnormalized row, a sentence silently truncated
at the encoder's max length, or a cached ``.npy`` that no longer matches the text
it was keyed to would all still produce plausible numbers. So check them.

  1  BACKEND    a real sentence-transformer loaded, not the hashing fallback
  2  NORM       rows are L2-normalized, so a dot product IS cosine
  3  DEGENERATE no zero vectors, no empty texts
  4  COLLISION  distinct sentences that land on identical vectors
  5  TRUNCATION sentences longer than the encoder's max_seq_length
  6  CACHE      a sample re-encoded from scratch equals the memoized vectors
  7  PREFIX     BGE v1.5 was trained with an instruction on the QUERY side.
                ``corpus_dump_vs_cell`` sends the bare question. This measures
                what that costs -- gold-cell rank and OSC with and without --
                on the frozen population, retrieval only, no reader.

Check 7 is the only one that costs real time; ``--skip-prefix`` drops it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus_dump_vs_cell import (ALPHA, FROZEN_POP, RHB_POPS, _CachedEncoder,
                                 aitqa_corpus, hitab_corpus, positions,
                                 realhitbench_corpus)
from error_worksheet import render_cells
from rag_agent.retrieve.encoders import (HashingEncoder, default_encoder,
                                         default_prefixes)
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax
from rag_agent.serialization.base import Chunk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "aitqa", "realhitbench"])
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--cell-scheme", default="S3c")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--retriever", default="dense", choices=list(ALPHA))
    ap.add_argument("--sample", type=int, default=256,
                    help="how many cells to re-encode for the cache check")
    ap.add_argument("--skip-prefix", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.dataset == "hitab":
        C = hitab_corpus(args.data_dir, args.split, args.population)
    elif args.dataset == "aitqa":
        C = aitqa_corpus()
    else:
        pop = args.population if args.population in RHB_POPS \
            else FROZEN_POP["realhitbench"]
        C = realhitbench_corpus(pin=pop, **RHB_POPS[pop])
    render_cells(C, args.cell_scheme, False)
    texts = list(C.cell_text)

    inner = default_encoder(model_name=args.embed_model)
    enc = _CachedEncoder(inner, args.cache_dir,
                         f"{args.dataset}_{args.split}_{args.embed_model}")
    emb = enc.encode(texts)
    # the population NAME is a command-line default for the two corpora that
    # carry exactly one frozen list, so echo what was actually built
    pop_name = (args.population if args.dataset == "hitab"
                else FROZEN_POP[args.dataset])
    rep = {"dataset": args.dataset, "population": pop_name,
           "cell_scheme": args.cell_scheme, "n_cells": len(texts),
           "embed_model": args.embed_model}

    # 1 BACKEND ------------------------------------------------------------
    rep["backend"] = {
        "encoder_name": inner.name,
        "class": type(inner).__name__,
        "is_hashing_fallback": isinstance(inner, HashingEncoder),
        "device": getattr(inner, "device", "?"),
        "dim": int(emb.shape[1]),
    }
    # 2 NORM ---------------------------------------------------------------
    norms = np.linalg.norm(emb, axis=1)
    rep["norm"] = {"min": float(norms.min()), "max": float(norms.max()),
                   "max_abs_dev_from_1": float(np.abs(norms - 1).max()),
                   "ok": bool(np.abs(norms - 1).max() < 1e-3)}
    # 3 DEGENERATE ---------------------------------------------------------
    rep["degenerate"] = {
        "zero_vectors": int((norms < 1e-8).sum()),
        "empty_texts": int(sum(1 for t in texts if not (t or "").strip())),
        "nan_or_inf": int((~np.isfinite(emb)).any(axis=1).sum()),
    }
    # 4 COLLISION ----------------------------------------------------------
    # distinct sentences landing on the same vector: the address collision this
    # repo tracks lexically, seen in the space the retriever actually scores in
    uniq_txt = len(set(texts))
    _, inv = np.unique(np.round(emb, 5), axis=0, return_inverse=True)
    rep["collision"] = {
        "unique_texts": uniq_txt,
        "duplicate_text_share": round(1 - uniq_txt / len(texts), 4),
        "unique_vectors_at_1e-5": int(inv.max()) + 1,
        "distinct_texts_sharing_a_vector":
            int(uniq_txt - (int(inv.max()) + 1)) if uniq_txt >= int(inv.max()) + 1 else 0,
    }
    # 5 TRUNCATION ---------------------------------------------------------
    trunc = {"checked": False}
    tokzr = getattr(getattr(inner, "model", None), "tokenizer", None)
    if tokzr is not None:
        maxlen = int(getattr(inner.model, "max_seq_length", 512))
        lens = np.array([len(tokzr(t, add_special_tokens=True)["input_ids"])
                         for t in texts], dtype=np.int32)
        trunc = {"checked": True, "max_seq_length": maxlen,
                 "p50": int(np.percentile(lens, 50)),
                 "p99": int(np.percentile(lens, 99)),
                 "max": int(lens.max()),
                 "n_truncated": int((lens > maxlen).sum()),
                 "share_truncated": round(float((lens > maxlen).mean()), 5)}
    rep["truncation"] = trunc
    # 6 CACHE --------------------------------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(texts), size=min(args.sample, len(texts)), replace=False)
    fresh = inner.encode([texts[i] for i in idx])
    delta = np.abs(fresh - emb[idx]).max()
    rep["cache"] = {"sampled": int(len(idx)), "max_abs_delta": float(delta),
                    "ok": bool(delta < 1e-4)}
    # 7 PREFIX -------------------------------------------------------------
    if not args.skip_prefix:
        qpre, ppre = default_prefixes(inner.name)
        chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=txt,
                        scheme=args.cell_scheme, kind="cell")
                  for txt, (t, i, j) in zip(texts, C.cell_owner)]
        ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
        alpha = ALPHA[args.retriever]
        pos_of_cell = {k: n for n, k in enumerate(C.cell_owner)}

        def sweep(prefix: str) -> dict:
            osc, ranks, r1 = [], [], []
            for q in C.queries:
                question = prefix + q["question"]
                bm = ix._bm25_scores(q["question"])   # BM25 reads the bare text
                dn = ix._dense_scores(question)
                fused = (dn if alpha == 1.0 else
                         alpha * _minmax(dn) + (1 - alpha) * _minmax(bm))
                rank = positions(list(np.argsort(-fused)))
                gp = [pos_of_cell[g] for g in q["gold_cells"] if g in pos_of_cell]
                if not gp:
                    continue
                rr = [int(rank[p]) + 1 for p in gp]
                ranks.append(float(np.mean(rr)))
                r1.append(int(min(rr) == 1))
                osc.append(int(max(rr) <= 60))   # a fixed prefix, budget-free
            return {"n": len(osc), "mean_gold_rank": round(float(np.mean(ranks)), 2),
                    "median_gold_rank": float(np.median(ranks)),
                    "top1_is_gold": round(float(np.mean(r1)), 4),
                    "all_gold_in_top60": round(float(np.mean(osc)), 4)}

        rep["prefix"] = {
            "trained_query_prefix": qpre,
            "trained_passage_prefix": ppre,
            "driver_applies_it": False,   # corpus_dump_vs_cell encodes q["question"] bare
            "bare": sweep(""),
            "with_prefix": sweep(qpre) if qpre else None,
        }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
