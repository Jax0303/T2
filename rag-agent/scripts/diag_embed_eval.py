#!/usr/bin/env python3
"""진단 Phase 1 — BGE-small 임베딩 + 표 단위 검색평가 (전처리×복잡도).

각 데이터셋(flat/hier) × 조건(C0/C1/C2[/C3])에 대해:
  표 직렬화 텍스트 임베딩 → 코퍼스 인덱스 → 질문 임베딩 → top-k → R@1/5/10/MRR/nDCG.
조건 간 비교는 C0 기준 paired bootstrap(질문단위 재표집, seed=42, B=10000)으로 R@1 차이 95%CI.

사용:
  python scripts/diag_embed_eval.py --dataset flat --split test --max-queries 2000
  python scripts/diag_embed_eval.py --dataset hier --split dev
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "
EMB_DIR = Path("diag/emb_cache")


def load_records(path):
    ids, texts = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        ids.append(r["table_id"]); texts.append(r["text"])
    return ids, texts


def embed(model, texts, tag, batch=256):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    cache = EMB_DIR / f"{tag}.npy"
    if cache.exists():
        return np.load(cache)
    emb = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(cache, emb.astype(np.float32))
    return emb.astype(np.float32)


def topk_ids(q_emb, corpus_emb, corpus_ids, k=20):
    qd = torch.tensor(q_emb, device="cuda" if torch.cuda.is_available() else "cpu")
    cd = torch.tensor(corpus_emb, device=qd.device).T  # d × N
    out = []
    for i in range(0, qd.shape[0], 512):
        sims = qd[i:i+512] @ cd                       # b × N (normalized → cosine)
        idx = torch.topk(sims, k, dim=1).indices.cpu().numpy()
        for row in idx:
            out.append([corpus_ids[j] for j in row])
    return out


def metrics(ranked, golds):
    def r_at_k(ids, g, k): return int(g in ids[:k])
    def mrr(ids, g):
        for i, x in enumerate(ids, 1):
            if x == g: return 1.0 / i
        return 0.0
    def ndcg(ids, g, k=10):
        for i, x in enumerate(ids[:k], 1):
            if x == g: return 1.0 / math.log2(i + 1)
        return 0.0
    r1 = [r_at_k(a, g, 1) for a, g in zip(ranked, golds)]
    return {
        "R@1": float(np.mean(r1)),
        "R@5": float(np.mean([r_at_k(a, g, 5) for a, g in zip(ranked, golds)])),
        "R@10": float(np.mean([r_at_k(a, g, 10) for a, g in zip(ranked, golds)])),
        "MRR": float(np.mean([mrr(a, g) for a, g in zip(ranked, golds)])),
        "nDCG@10": float(np.mean([ndcg(a, g) for a, g in zip(ranked, golds)])),
        "_r1_vec": r1,
    }


def paired_bootstrap(a, b, B=10000, seed=42):
    """R@1 차이(b-a) 95%CI + P(diff!=0)."""
    a, b = np.array(a), np.array(b)
    diffs = b - a
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(diffs.mean()), "ci95": [float(lo), float(hi)],
            "sig": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["flat", "hier"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--conditions", default="C0,C1,C2")
    args = ap.parse_args()

    base = Path(f"diag/{args.dataset}")
    queries = [json.loads(l) for l in open(base / "queries.jsonl", encoding="utf-8")]
    queries = [q for q in queries if q["split"] == args.split]
    if args.max_queries and len(queries) > args.max_queries:
        random.seed(42); queries = random.sample(queries, args.max_queries)
    q_texts = [QUERY_INSTR + q["question"] for q in queries]
    golds = [q["gold_table_id"] for q in queries]
    print(f"[{args.dataset}/{args.split}] queries={len(queries)}")

    model = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    q_emb = embed(model, q_texts, f"{args.dataset}_q_{args.split}_{len(queries)}")

    conditions = args.conditions.split(",")
    results, r1_vecs = {}, {}
    for c in conditions:
        ids, texts = load_records(base / "serialized" / f"{c}.records.jsonl")
        cemb = embed(model, texts, f"{args.dataset}_{c}")
        ranked = topk_ids(q_emb, cemb, ids, k=20)
        m = metrics(ranked, golds)
        r1_vecs[c] = m.pop("_r1_vec")
        results[c] = m
        print(f"  {c}: R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} MRR={m['MRR']:.3f}")

    # C0 기준 paired bootstrap
    boot = {}
    if "C0" in r1_vecs:
        for c in conditions:
            if c == "C0":
                continue
            boot[f"{c}-C0"] = paired_bootstrap(r1_vecs["C0"], r1_vecs[c])
            d = boot[f"{c}-C0"]
            print(f"  Δ{c}-C0 R@1={d['delta']:+.3f} CI{d['ci95']} sig={d['sig']}")

    out = {"dataset": args.dataset, "split": args.split, "n_queries": len(queries),
           "n_corpus": len(ids), "model": MODEL, "metrics": results, "bootstrap_vs_C0": boot}
    outpath = Path("results") / f"diag_{args.dataset}_{args.split}.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"[saved] {outpath}")


if __name__ == "__main__":
    main()
