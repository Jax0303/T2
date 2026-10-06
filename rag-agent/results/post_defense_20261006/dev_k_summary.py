"""PREREG-2026-10-06-post-defense.md A: dev 332 k 사다리에 k=1·3 을 더해 공식 EM·형식 허용 정답 수와 k=20 대비 McNemar(정확) p 를 적는다."""
import json, sys
from math import comb
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from mh_arms import official_answers                         # noqa: E402
from rag_agent.eval.multihiertt_em import format_match, mh_exact_match   # noqa: E402

OLD, NEW = Path("results/mh_only_20260928/dev_k"), Path("results/post_defense_20261006/dev_k")
gold = official_answers("validation")
def load(k):
    p = (NEW if k in (1, 3) else OLD) / f"dev_s3c_k{k}_answer.jsonl"
    if not p.exists(): return None
    rows = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); prog = r["kind"] == "arith"
        rows[r["query_id"]] = (int(mh_exact_match(r["pred"], gold[r["query_id"]], prog)), int(format_match(r["pred"], gold[r["query_id"]], prog)))
    return rows
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
ref = load(20)
print("k | 문항 | 공식 EM | 형식 허용 | 허용 b:c 대 k=20 | p")
for k in (1, 3, 5, 10, 15, 20, 30, 50, 75):
    rows = load(k)
    if rows is None or len(rows) != len(ref): print(f"{k} | (없음 또는 진행 중 {0 if rows is None else len(rows)}행)"); continue
    assert set(rows) == set(ref), k
    em = sum(v[0] for v in rows.values()); le = sum(v[1] for v in rows.values())
    b = sum(rows[q][1] and not ref[q][1] for q in rows); c = sum(ref[q][1] and not rows[q][1] for q in rows)
    print(f"{k} | {len(rows)} | {em} | {le} | {b}:{c} | {mcnemar(b, c):.4f}")
