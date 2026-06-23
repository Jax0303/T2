#!/usr/bin/env python3
"""교수님 가설(표 단위): 라벨링한 메타데이터에 쿼리를 매핑하면 *전체 코퍼스*에서 맞는 표를
오류 없이 찾는가? — LLM 없이 BM25(어휘 매칭)로만 측정.

기존 DIAGNOSIS(diag_embed_eval)는 dense 임베딩(BGE) 기반이었다. 여기선 임베더조차 빼고
순수 BM25로, "정보 증강(메타데이터/스키마 라벨링)"이 표 검색 recall을 올리는지를 모델 무관하게 본다.

조건(인덱싱 텍스트) = 커밋된 직렬화 산출물:
  C0  표 본문만(raw)
  C1  C0 + 제목/섹션/캡션(metadata)
  C2  C1 + 스키마/헤더경로(schema)        ← 교수님이 말한 구조 메타데이터에 해당

지표: recall@k = gold 표가 top-k 안에 든 쿼리 비율, MRR. 검색 단위 = 표 전체(table-level).
사용: python3 scripts/corpus_table_recall.py --split test --n 2000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
KS = (1, 3, 5, 10, 20, 50)
_TOK_RE = re.compile(r"[a-z0-9]+")


def tok(text: str):
    return _TOK_RE.findall(text.lower())


def load_corpus(path: Path):
    """직렬화 records jsonl → (table_ids, texts)."""
    ids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids.append(d["table_id"])
            texts.append(d.get("text", ""))
    return ids, texts


def load_queries(path: Path, split: str, n: int):
    qs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if split and d.get("split") != split:
                continue
            qs.append(d)
    rng = random.Random(SEED)
    rng.shuffle(qs)
    return qs[:n] if n else qs


def eval_condition(name, corpus_path, queries):
    from rank_bm25 import BM25Okapi
    ids, texts = load_corpus(corpus_path)
    id2idx = {t: i for i, t in enumerate(ids)}
    t0 = time.time()
    bm25 = BM25Okapi([tok(t) for t in texts])
    build_s = time.time() - t0

    # gold 표가 코퍼스에 있는 쿼리만 평가 (공정: 검색 대상에 정답이 존재)
    qs = [q for q in queries if q["gold_table_id"] in id2idx]
    recall = {k: 0 for k in KS}
    rr_sum = 0.0
    t1 = time.time()
    for q in qs:
        gold_idx = id2idx[q["gold_table_id"]]
        scores = bm25.get_scores(tok(q["question"]))
        # gold 표보다 점수 높은 표 수 = gold 의 0-base 순위
        gold_score = scores[gold_idx]
        rank = int((scores > gold_score).sum())          # 동점은 gold 우위로 보수적
        for k in KS:
            if rank < k:
                recall[k] += 1
        rr_sum += 1.0 / (rank + 1)
    n = len(qs)
    out = {
        "condition": name,
        "corpus": str(corpus_path.relative_to(ROOT)),
        "n_tables": len(ids),
        "n_eval": n,
        "recall@k": {str(k): round(recall[k] / n, 4) for k in KS} if n else {},
        "MRR": round(rr_sum / n, 4) if n else None,
        "build_s": round(build_s, 1),
        "eval_s": round(time.time() - t1, 1),
    }
    print(f"[{name}] tables={len(ids)} n={n}  "
          f"R@1={out['recall@k'].get('1')} R@5={out['recall@k'].get('5')} "
          f"R@10={out['recall@k'].get('10')} MRR={out['MRR']}  "
          f"(build {out['build_s']}s eval {out['eval_s']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="diag/flat/serialized")
    ap.add_argument("--queries", default="diag/flat/queries.jsonl")
    ap.add_argument("--conditions", default="C0,C1,C2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/corpus_table_recall.json")
    args = ap.parse_args()

    queries = load_queries(ROOT / args.queries, args.split, args.n)
    print(f"[corpus_table_recall] split={args.split} sampled={len(queries)} "
          f"conditions={args.conditions} (BM25, LLM-free)", flush=True)

    results = {}
    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        cpath = ROOT / args.corpus_dir / f"{cond}.records.jsonl"
        if not cpath.exists():
            print(f"  skip {cond}: {cpath} not found", flush=True)
            continue
        results[cond] = eval_condition(cond, cpath, queries)

    summary = {
        "config": {"retriever": "bm25", "llm": None, "split": args.split,
                   "n_requested": args.n, "seed": SEED,
                   "unit": "table", "ks": list(KS)},
        "results": results,
    }
    # 메타데이터/스키마 라벨링 이득 (델타)
    if "C0" in results and "C1" in results:
        summary["delta_C1_C0_R@1"] = round(
            results["C1"]["recall@k"]["1"] - results["C0"]["recall@k"]["1"], 4)
    if "C1" in results and "C2" in results:
        summary["delta_C2_C1_R@1"] = round(
            results["C2"]["recall@k"]["1"] - results["C1"]["recall@k"]["1"], 4)

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== corpus-level table recall (BM25, no LLM) ===", flush=True)
    print(json.dumps({k: v.get("recall@k") for k, v in results.items()},
                     ensure_ascii=False, indent=2), flush=True)
    for key in ("delta_C1_C0_R@1", "delta_C2_C1_R@1"):
        if key in summary:
            print(f"{key} = {summary[key]:+.4f}", flush=True)
    print(f"→ {outp}", flush=True)


if __name__ == "__main__":
    main()
