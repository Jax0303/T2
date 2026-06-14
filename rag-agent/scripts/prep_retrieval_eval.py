#!/usr/bin/env python3
"""Pre-retrieval preprocessing experiment: condition × retriever × dataset.

Measures how cumulative preprocessing of the *indexed text* (C0 raw →
C1 +metadata → C2 +schema description → C3 +synthetic questions; C2h =
hierarchical header-path variant) changes table-retrieval recall, on a flat
corpus (OpenWikiTable) and a hierarchical corpus (HiTab).

Examples
--------
# BM25, flat corpus, 1k test queries (CPU-only, no model download):
python rag-agent/scripts/prep_retrieval_eval.py \
    --dataset owt --retriever bm25 \
    --conditions C0,C1,C2,C3 --synth template \
    --n-queries 1000 --out rag-agent/results/prep/owt_bm25.json

# Dense BGE-small (local GPU machine):
python rag-agent/scripts/prep_retrieval_eval.py \
    --dataset owt --retriever dense --model BAAI/bge-small-en-v1.5 \
    --device cuda --conditions C0,C1,C2,C3 --synth template \
    --n-queries 1000 --out rag-agent/results/prep/owt_dense.json

# HiTab (hierarchical), with the header-path condition:
python rag-agent/scripts/prep_retrieval_eval.py \
    --dataset hitab --data-dir rag-agent/data/hitab --retriever dense \
    --conditions C0,C1,C2,C2h,C3 --synth template \
    --out rag-agent/results/prep/hitab_dense.json

Outputs a JSON with per-condition metrics (R@1/5/10, MRR), paired-bootstrap
deltas (vs C0 and vs the previous rung of the ladder), and per-query gold
ranks for offline re-analysis.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.prep.conditions import (  # noqa: E402
    CONDITIONS, PrepTable, from_hitab, from_openwikitable, serialize,
)
from rag_agent.prep.stats import (  # noqa: E402
    paired_delta_bootstrap, summarize_condition,
)
from rag_agent.prep.synth import build_synth  # noqa: E402

TOP_N = 100  # ranks deeper than this are recorded as None


# ---- dataset loading --------------------------------------------------------

def load_owt(data_dir: Path, split: str):
    tables = []
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            tables.append(from_openwikitable(json.loads(line)))
    queries = []
    with open(data_dir / f"queries_{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            queries.append((r["question_id"], r["question"], r["gold_table_id"]))
    return tables, queries


def load_hitab(data_dir: Path, split: str):
    from rag_agent.data.loader import load_samples, load_table
    from rag_agent.stores.original_store import build_original_table

    # full 3,597-table corpus (both hmt subdirs handled by load_table)
    table_dir = None
    for cand in [data_dir / "data" / "tables", data_dir / "data" / "tables" / "tables"]:
        if cand.exists():
            table_dir = cand
            break
    if table_dir is None:
        raise FileNotFoundError(f"no tables dir under {data_dir}")
    ids = sorted({p.stem for sub in ("hmt", "raw") if (table_dir / sub).exists()
                  for p in (table_dir / sub).glob("*.json")})
    tables = []
    for tid in ids:
        raw = load_table(tid, str(data_dir))
        if raw is not None:
            tables.append(from_hitab(build_original_table(raw)))

    samples = load_samples(str(data_dir), split=split)
    queries = [(s["id"], s["question"], s["table_id"]) for s in samples]
    return tables, queries


# ---- retriever backends -----------------------------------------------------

_TOKEN_SPLIT = __import__("re").compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return _TOKEN_SPLIT.findall(text.lower())


class BM25Retriever:
    def __init__(self, texts):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def rank_gold(self, query: str, gold_idx: int):
        import numpy as np

        scores = self.bm25.get_scores(_tokenize(query))
        gold_score = scores[gold_idx]
        # rank = 1 + number of docs strictly above the gold (ties favor gold,
        # consistent across conditions so comparisons stay fair)
        rank = int(np.sum(scores > gold_score)) + 1
        return rank if rank <= TOP_N else None


class DenseRetriever:
    def __init__(self, texts, model_name: str, device: str, batch_size: int):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_emb = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )

    def rank_gold(self, query: str, gold_idx: int):
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
        scores = self.doc_emb @ q
        gold_score = scores[gold_idx]
        rank = int(self.np.sum(scores > gold_score)) + 1
        return rank if rank <= TOP_N else None


def build_retriever(kind, texts, args):
    if kind == "bm25":
        return BM25Retriever(texts)
    if kind == "dense":
        return DenseRetriever(texts, args.model, args.device, args.batch_size)
    raise ValueError(kind)


# ---- main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["owt", "hitab"], required=True)
    p.add_argument("--data-dir", default=None,
                   help="default: rag-agent/data/openwikitable or rag-agent/data/hitab")
    p.add_argument("--split", default=None,
                   help="owt: test|valid (default test); hitab: dev|test (default dev)")
    p.add_argument("--conditions", default="C0,C1,C2,C3")
    p.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-rows", type=int, default=20,
                   help="rows serialized per table (same across conditions)")
    p.add_argument("--n-queries", type=int, default=1000,
                   help="random query subsample (0 = all)")
    p.add_argument("--max-corpus", type=int, default=0,
                   help="cap corpus size to this many tables (0 = full). The "
                        "gold tables of the sampled queries are always kept; "
                        "the remainder is filled with seeded random distractors. "
                        "Lets an LLM-synth (C3) ablation run on a feasible "
                        "number of tables without changing the query set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", default="1,5,10")
    p.add_argument("--boot-iters", type=int, default=10000)
    p.add_argument("--synth", default="template",
                   help="C3 question source: 'template' or 'llm:<spec>'")
    p.add_argument("--synth-n", type=int, default=5)
    p.add_argument("--synth-cache", default=None,
                   help="jsonl cache for llm synth (reused across conditions/runs)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            p.error(f"unknown condition {c}; choose from {CONDITIONS}")
    ks = [int(k) for k in args.ks.split(",")]

    base = Path(__file__).resolve().parents[1]
    if args.dataset == "owt":
        data_dir = Path(args.data_dir or base / "data" / "openwikitable")
        tables, queries = load_owt(data_dir, args.split or "test")
    else:
        data_dir = Path(args.data_dir or base / "data" / "hitab")
        tables, queries = load_hitab(data_dir, args.split or "dev")

    idx_by_id = {t.table_id: i for i, t in enumerate(tables)}
    queries = [(qid, q, g) for qid, q, g in queries if g in idx_by_id]
    if args.n_queries and len(queries) > args.n_queries:
        rng = random.Random(args.seed)
        queries = rng.sample(queries, args.n_queries)

    # Optionally cap the corpus: keep every gold table the sampled queries
    # point at, then add seeded random distractors up to --max-corpus. The
    # query set is untouched, so a slow LLM-synth condition becomes tractable.
    if args.max_corpus and len(tables) > args.max_corpus:
        gold_ids = {g for _, _, g in queries}
        keep = set(gold_ids)
        if len(keep) > args.max_corpus:
            raise SystemExit(
                f"--max-corpus {args.max_corpus} < {len(keep)} gold tables; "
                f"lower --n-queries or raise --max-corpus")
        distractors = [t.table_id for t in tables if t.table_id not in keep]
        random.Random(args.seed).shuffle(distractors)
        for tid in distractors:
            if len(keep) >= args.max_corpus:
                break
            keep.add(tid)
        tables = [t for t in tables if t.table_id in keep]
        idx_by_id = {t.table_id: i for i, t in enumerate(tables)}

    print(f"corpus: {len(tables)} tables | queries: {len(queries)} "
          f"| conditions: {conditions} | retriever: {args.retriever}")

    synth = build_synth(args.synth, n_questions=args.synth_n,
                        cache_path=args.synth_cache) if "C3" in conditions else None

    results = {
        "config": {k: v for k, v in vars(args).items()},
        "n_tables": len(tables),
        "n_queries": len(queries),
        "conditions": {},
        "deltas": {},
        "per_query": {"question_ids": [qid for qid, _, _ in queries]},
    }

    ranks_by_cond = {}
    for cond in conditions:
        t0 = time.time()
        texts = [serialize(t, cond, max_rows=args.max_rows, synth_provider=synth)
                 for t in tables]
        t_ser = time.time() - t0
        retriever = build_retriever(args.retriever, texts, args)
        t_idx = time.time() - t0 - t_ser

        ranks = []
        for qid, question, gold in queries:
            ranks.append(retriever.rank_gold(question, idx_by_id[gold]))
        ranks_by_cond[cond] = ranks

        summary = summarize_condition(ranks, ks)
        summary["serialize_s"] = round(t_ser, 1)
        summary["index_s"] = round(t_idx, 1)
        summary["query_s"] = round(time.time() - t0 - t_ser - t_idx, 1)
        results["conditions"][cond] = summary
        results["per_query"][cond] = ranks
        print(f"[{cond}] " + " ".join(f"{k}={v}" for k, v in summary.items()))

    # paired bootstrap: each condition vs C0, and vs the previous rung
    baseline = conditions[0]
    for i, cond in enumerate(conditions[1:], start=1):
        for ref in {baseline, conditions[i - 1]}:
            if ref == cond:
                continue
            key = f"{cond} - {ref}"
            results["deltas"][key] = {}
            for k in ks:
                mean, lo, hi = paired_delta_bootstrap(
                    ranks_by_cond[cond], ranks_by_cond[ref], k,
                    n_iters=args.boot_iters, seed=args.seed,
                )
                sig = lo > 0 or hi < 0
                results["deltas"][key][f"R@{k}"] = {
                    "delta": round(mean, 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "significant": sig,
                }
                mark = " *" if sig else ""
                print(f"Δ {key} R@{k}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
