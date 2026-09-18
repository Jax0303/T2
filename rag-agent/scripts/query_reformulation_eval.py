#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Priority (c) of the 2026-09-17 post-leaf-repeat investigation ([[leaf-repeat-
axis-closed-2026-09-17]]): does rewriting the question into a HyDE-style
hypothetical table-fact sentence, THEN encoding that instead of the raw
question, move R@1 past the ~.62 ceiling every structural_leaf text/encoder
variant hit? (a) selector/reader swap was invalid, (b) encoder swap made it
worse -- this is the one untried idea left in that session's priority list.

Isolates the dense side only, same convention as the leaf-repeat channel
ablation (results/embedding_fusion_diagnosis_20260917/leaf_channel_ablation.json):
BM25 stays pinned to the raw question text; only the dense query vector comes
from the LLM's rewrite. Text (structural_leaf), encoder (BAAI/bge-base-en-v1.5)
and hybrid formula (alpha=0.7) all held fixed at their deployed values.

n=150 pilot (first 150 of the primary population, sorted by query_id -- the
same subset convention scripts/selector_top20_eval.py's pilots use), because
each query costs one LLM call. Compared by McNemar against the deployed
structural_leaf hybrid baseline's per-query hits (results/sentence_disambiguation_20260916/
retrieval_structural_leaf.jsonl), restricted to the same 150 query_ids.

  PYTHONPATH=. .venv/bin/python scripts/query_reformulation_eval.py
  PYTHONPATH=. .venv/bin/python scripts/query_reformulation_eval.py --limit 20 --reader groq:llama-3.1-8b-instant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402
from scipy.stats import binomtest                                     # noqa: E402

from rag_agent.eval.artifacts import digest                           # noqa: E402
from rag_agent.llm.factory import build_llm                           # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize        # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from bottleneck_diagnosis import load_primary_population              # noqa: E402
from sentence_disambiguation_eval import build_leaf_corpus, encode_corpus  # noqa: E402

OUT_DIR = ROOT / "results/query_reformulation_20260917"
BASELINE_HITS = ROOT / "results/sentence_disambiguation_20260916/retrieval_structural_leaf.jsonl"
CACHE = ROOT / ".cache/query_reformulation"
ALPHA = 0.7

REFORMULATE_SYSTEM = (
    "Rewrite the question as one short hypothetical fact sentence, the way a "
    "cell in a statistical table would be described: "
    '"For <category/breakdown>, <another breakdown if any>, <value>." '
    "Guess a plausible number or category for the value if you don't know the "
    "real one -- a specific wrong guess is fine, a hedge or refusal is not. "
    "Output ONLY the sentence, nothing else."
)


def pilot_subset(pop: dict, limit: int) -> list:
    """First ``limit`` of the population sorted by query_id (0 = all)."""
    ordered = sorted(pop.items())
    return ordered[:limit] if limit else ordered


def baseline_hits(qids: set) -> dict:
    out = {}
    with BASELINE_HITS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["query_id"] in qids:
                out[r["query_id"]] = bool(r["esm"])
    return out


def mcnemar(a: dict, b: dict, label: str) -> dict:
    qids = sorted(set(a) & set(b))
    a_only = sum(a[q] and not b[q] for q in qids)
    b_only = sum(b[q] and not a[q] for q in qids)
    both = sum(a[q] and b[q] for q in qids)
    n_disc = a_only + b_only
    p = binomtest(min(a_only, b_only), n_disc, 0.5).pvalue if n_disc else 1.0
    out = {"label": label, "n": len(qids),
           "a_acc": round(sum(a.values()) / len(a), 4) if a else None,
           "b_acc": round(sum(b.values()) / len(b), 4) if b else None,
           "a_only": a_only, "b_only": b_only, "both": both,
           "discordant": n_disc, "p_value": round(p, 4)}
    print(f"{label}: n={out['n']} a_acc={out['a_acc']} b_acc={out['b_acc']} "
         f"a_only={a_only} b_only={b_only} both={both} discordant={n_disc} p={p:.4f}")
    return out


def reformulate_all(llm, ordered: list, cache_path: Path) -> dict:
    """question -> hypothetical fact sentence, one LLM call per query_id, cached."""
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for qid, r in ordered:
        if qid in cached:
            continue
        text = llm.complete(REFORMULATE_SYSTEM, r["question"], max_tokens=80)
        cached[qid] = text.strip() or r["question"]  # empty completion: fall back to the raw question
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
    return cached


def run(data_dir: str = "data/hitab", limit: int = 150,
       reader: str = "local:Qwen/Qwen3-8B?quantization=4bit") -> dict:
    pop = load_primary_population()
    ordered = pilot_subset(pop, limit)
    texts, coords = build_leaf_corpus(data_dir, "structural_leaf")
    coord_index = {c: idx for idx, c in enumerate(coords)}
    bm = SparseBM25(_tokenize(t) for t in texts)

    enc = default_encoder()
    emb, cache_hit = encode_corpus(texts, enc)

    llm = build_llm(reader)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_key = digest([qid for qid, _ in ordered])[:16]
    reformulated = reformulate_all(llm, ordered, CACHE / f"{reader.split(':')[0]}_{cache_key}.json")

    rows, hits = [], {}
    for qid, r in ordered:
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            continue
        dense = emb @ enc.encode_query([reformulated[qid]])[0].astype(np.float32)
        sparse = bm.get_scores(_tokenize(r["question"]))  # sparse channel stays on the raw question
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
        order = np.argsort(-hybrid, kind="stable")
        gold_rank = int(np.where(order == gidx)[0][0]) + 1
        hit = gold_rank == 1
        hits[qid] = hit
        rows.append({"query_id": qid, "question": r["question"],
                    "reformulated": reformulated[qid], "gold_rank": gold_rank, "esm": int(hit)})

    recall_at_1 = round(sum(hits.values()) / len(hits), 4) if hits else None
    comparison = mcnemar(baseline_hits(set(hits)), hits,
                         "structural_leaf deployed vs query_reformulation(dense-only)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"n": len(hits), "recall_at_1": recall_at_1, "embed_cache_hit": cache_hit,
              "reader": reader, "alpha": ALPHA, "comparison_vs_deployed": comparison}
    (OUT_DIR / "query_reformulation_eval.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "query_reformulation_eval_records.jsonl").open(
            "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--limit", type=int, default=150, help="0 = full n=991 population")
    ap.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit",
                    help="LLM that rewrites the question, e.g. groq:llama-3.1-8b-instant")
    a = ap.parse_args()
    run(a.data_dir, a.limit, a.reader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
