#!/usr/bin/env python3
"""Retrieval-only 평가 — LLM 없이 검색 우위만 정량화.

thesis: 원본 구조검색(헤더+숫자)이 직렬화 dense VDB(DTR/DPR 계열)를 이긴다.
검색지표(R@1/R@5/MRR/nDCG@10)는 생성형 LLM이 필요 없다. 같은 후보 풀
(HiTab dev 고유 테이블, DTR R@k 프로토콜) 위에서 4개 검색기를 쌍대 비교한다.

  structural_full   : 헤더 토큰 0.6 + 숫자 셀 0.4   (제안)
  structural_h0     : 헤더 토큰만 (w_num=0)          (숫자 누수 제거)
  keyword           : 전체 토큰(셀 포함) 키워드      (메커니즘 분리)
  vdb               : BGE-large dense 검색           (baseline, DPR/DTR 계열)

사용: HF_HOME=... HITAB_DIR=... CHROMA_DIR=... python scripts/retrieval_eval.py [--limit N] [--out f.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codegen_eval as ce  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_agent.eval.metrics import difficulty_class  # noqa: E402


def ranked_ids(hits):
    return [tid for tid, _ in hits]


def r_at_k(ids, gold, k):
    return int(gold in ids[:k])


def mrr(ids, gold):
    for i, x in enumerate(ids, 1):
        if x == gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ids, gold, k=10):
    for i, x in enumerate(ids[:k], 1):
        if x == gold:
            return 1.0 / math.log2(i + 1)
    return 0.0


def agg(rows, key):
    vals = [r[key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def paired_bootstrap(diffs, B=5000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(B):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return (sum(diffs) / n, means[int(0.025 * B)], means[int(0.975 * B)])


def mcnemar(a_hit, b_hit):
    # b improves: b=1,a=0 ; a better: a=1,b=0
    b01 = sum(1 for a, b in zip(a_hit, b_hit) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_hit, b_hit) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # exact two-sided binomial p (p=0.5)
    k = min(b01, b10)
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return b01, b10, min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="평가 쿼리 수 제한 (디버그)")
    ap.add_argument("--w-num", type=float, default=0.4)
    ap.add_argument("--topk", type=int, default=540, help="full ranking 위해 풀 크기로")
    ap.add_argument("--out", type=str, default="results/retrieval_eval.json")
    args = ap.parse_args()

    samples = ce.load_samples("dev")
    if args.limit:
        samples = samples[: args.limit]

    # ── 후보 풀 구성: dev 고유 gold 테이블 전부 ──
    print("OriginalDB 구축 중...")
    orig = ce.OriginalDB()
    seen = set()
    for s in samples:
        tid = ce.get_table_id(s)
        if tid and tid not in seen:
            raw = ce.load_table(tid)
            if raw:
                orig.add(raw)
                seen.add(tid)
    pool = set(orig._tables)
    print(f"  풀: {len(pool)} 테이블")

    print("VectorDB 로드 중 (BGE-large, CPU)...")
    vdb = ce.VectorDB(ce.CHROMA_DIR, device="cpu")

    # pool 중 VDB 벡터가 있는 테이블 커버리지 점검 (1회 쿼리로는 못 봄 → add 시점에 확인 불가,
    # 대신 평가 끝나고 vdb가 한 번이라도 랭크한 테이블 집합으로 근사)
    vdb_seen = set()

    rows = []
    miss_pool = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        q = ce.get_query(s)
        gold = ce.get_table_id(s)
        if gold not in pool:
            miss_pool += 1
            continue
        sf = ranked_ids(orig.structural_search(q, top_k=args.topk, w_num=args.w_num))
        sh = ranked_ids(orig.structural_search(q, top_k=args.topk, w_num=0.0))
        kw = ranked_ids(orig.keyword_search(q, top_k=args.topk))
        vd = ranked_ids(vdb.search(q, top_k=args.topk, allowed_ids=pool))
        vdb_seen.update(vd)
        row = {"cls": difficulty_class(s), "gold": gold}
        for name, ids in (("structural_full", sf), ("structural_h0", sh),
                          ("keyword", kw), ("vdb", vd)):
            row[name] = {
                "r1": r_at_k(ids, gold, 1), "r5": r_at_k(ids, gold, 5),
                "mrr": mrr(ids, gold), "ndcg": ndcg_at_k(ids, gold, 10),
            }
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)}  ({(time.time()-t0):.0f}s)")

    retrievers = ["structural_full", "structural_h0", "keyword", "vdb"]
    metrics = ["r1", "r5", "mrr", "ndcg"]

    def overall(ret, m):
        return sum(r[ret][m] for r in rows) / len(rows) if rows else 0.0

    print("\n" + "=" * 72)
    print(f"검색풀 {len(pool)} 테이블 | 평가쿼리 {len(rows)} (풀밖 gold {miss_pool} 제외)")
    print(f"VDB가 랭크한 풀 테이블 수: {len(vdb_seen & pool)}/{len(pool)}")
    print("=" * 72)
    hdr = f"{'retriever':<18}" + "".join(f"{m.upper():>9}" for m in ["R@1", "R@5", "MRR", "nDCG@10"])
    print(hdr)
    for ret in retrievers:
        line = f"{ret:<18}"
        for m in metrics:
            line += f"{overall(ret, m):>9.3f}"
        print(line)

    # ── 쌍대 비교 ──
    print("\n쌍대 비교 (B - A, 95% paired-bootstrap CI; * = CI가 0 제외)")
    pairs = [("vdb", "structural_full"), ("keyword", "structural_full"),
             ("structural_h0", "structural_full")]
    comparisons = {}
    for a, b in pairs:
        print(f"\n  A={a}  →  B={b}")
        comp = {}
        for m in metrics:
            diffs = [r[b][m] - r[a][m] for r in rows]
            mean, lo, hi = paired_bootstrap(diffs)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"    Δ{m.upper():<8} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}")
            comp[m] = {"delta": mean, "ci": [lo, hi], "sig": sig.strip() == "*"}
        b01, b10, p = mcnemar([r[a]["r1"] for r in rows], [r[b]["r1"] for r in rows])
        print(f"    McNemar R@1: B>A={b01} A>B={b10} p={p:.4g}")
        comp["mcnemar_r1"] = {"b_better": b01, "a_better": b10, "p": p}
        comparisons[f"{a}_vs_{b}"] = comp

    # ── per-class (제안 vs baseline R@1) ──
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    print("\nper-class R@1 (structural_full vs vdb), n")
    cls_out = {}
    for cls, rs in sorted(by_cls.items()):
        sf = sum(r["structural_full"]["r1"] for r in rs) / len(rs)
        vd = sum(r["vdb"]["r1"] for r in rs) / len(rs)
        print(f"  {cls:<22} sf={sf:.3f}  vdb={vd:.3f}  (n={len(rs)})")
        cls_out[cls] = {"structural_full_r1": sf, "vdb_r1": vd, "n": len(rs)}

    out = {
        "config": {
            "pool_size": len(pool), "n_eval": len(rows), "gold_out_of_pool": miss_pool,
            "vdb_pool_coverage": len(vdb_seen & pool) / len(pool) if pool else 0.0,
            "w_num": args.w_num, "topk": args.topk,
            "protocol": "DTR R@k (Herzig et al. NAACL 2021); HiTab dev; binary single-gold nDCG (Jarvelin & Kekalainen 2002)",
            "embedder": "BAAI/bge-large-en-v1.5", "collection": "plain_markdown_bge_large_en_v1_5",
        },
        "overall": {ret: {m: overall(ret, m) for m in metrics} for ret in retrievers},
        "paired": comparisons,
        "by_class": cls_out,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n저장: {outp}")


if __name__ == "__main__":
    main()
