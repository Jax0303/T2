"""질문 쪼개기 검색 대 기준 — 정답 셀 전부가 상위 m 안에 드는 문항 수(dev 395), 같은 문항 짝 McNemar. PREREG-2026-10-21-decompose-dev.md"""
import json
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
def load(paths):
    R = {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if "doc" in r: R[r["query_id"]] = r
    return R
old = load([ROOT / "results/mh_only_20260928/dev_k/dev_s3c_k100_records.jsonl", ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"])
base = load([HERE / "validation_base_k100_records.jsonl", HERE / "train_base_k100_records.jsonl"])
sub = load([HERE / "validation_sub_k100_records.jsonl", HERE / "train_sub_k100_records.jsonl"])
qs = sorted(base); N = len(qs); old = {q: old[q] for q in qs}
same = sum(old[q]["doc"]["ranked_units"][:20] == base[q]["doc"]["ranked_units"][:20] for q in qs)
print(f"문항 {N}; 기준 재실행이 예전 기록과 상위 20 순위 같음: {same}/{N}")
def hit(R, q, m): return set(R[q]["gold_ids"]) <= set(u[0] for u in R[q]["doc"]["ranked_units"][:m])
print("| 상위 m | 기준 | 쪼개기 | 차이 | 쪼개기만:기준만 | p |\n|---|---|---|---|---|---|")
res = {}
for m in (3, 5, 10, 20, 50, 100):
    hb = {q: hit(base, q, m) for q in qs}; hs = {q: hit(sub, q, m) for q in qs}
    b = sum(hs[q] and not hb[q] for q in qs); c = sum(hb[q] and not hs[q] for q in qs)
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    res[m] = {"base": sum(hb.values()), "sub": sum(hs.values()), "b": b, "c": c, "p": p}
    print(f"| {m} | {sum(hb.values())} ({100*sum(hb.values())/N:.1f}%) | {sum(hs.values())} ({100*sum(hs.values())/N:.1f}%) | {sum(hs.values())-sum(hb.values()):+d} | {b}:{c} | {p:.4f} |")
import collections
S = json.load(open(HERE / "sub_dev.json", encoding="utf-8")); n = [len(S[q]) for q in qs]
print(f"\n문구 수 평균 {sum(n)/N:.2f}, 분포 {dict(sorted(collections.Counter(n).items()))}")
for layer in sorted({base[q]["layer"] for q in qs}):
    L = [q for q in qs if base[q]["layer"] == layer]
    print(f"  {layer} {len(L)}: 상위 10 기준 {sum(hit(base,q,10) for q in L)} → 쪼개기 {sum(hit(sub,q,10) for q in L)}; 상위 20 {sum(hit(base,q,20) for q in L)} → {sum(hit(sub,q,20) for q in L)}")
json.dump({"n": N, "same_top20": same, "by_m": res}, open(HERE / "compare.json", "w"), indent=1)
