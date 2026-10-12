"""PREREG-2026-10-01-alpha-dev.md 집계: α 별 문서 안 검색 재현율@1/5/20, 전부 포함@20, 0.7 대비 McNemar."""
import json, sys
from pathlib import Path
from scipy.stats import binomtest

D = Path(__file__).parent
BASE = Path("results/mh_only_20260928/retrieval/mh_dev_s3c_b50_records.jsonl")
ALPHAS = ["0.0", "0.3", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]


def load(a):
    p = BASE if a == "0.7" else D / f"mh_dev_s3c_a{a}_records.jsonl"
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if "excluded" in r or "doc" not in r:
            continue
        gold = set(r["gold_ids"])
        units = r["doc"]["ranked_units"]
        rec = {}
        for k in (1, 5, 20):
            got = {c for u in units[:k] for c in u}
            rec[k] = len(gold & got) / len(gold)
        rec["all20"] = int(gold <= {c for u in units[:20] for c in u})
        out[r["query_id"]] = rec
    return out


runs = {a: load(a) for a in ALPHAS if a == "0.7" or (D / f"mh_dev_s3c_a{a}_records.jsonl").exists()}
ids = sorted(runs["0.7"])
for a, r in runs.items():
    assert sorted(r) == ids, a
print(f"n={len(ids)}")
print("alpha  R@1    R@5    R@20   all@20  b:c(vs .7)  p")
rows = []
for a in ALPHAS:
    if a not in runs:
        continue
    r = runs[a]
    m = {k: 100 * sum(r[q][k] for q in ids) / len(ids) for k in (1, 5, 20)}
    n_all = sum(r[q]["all20"] for q in ids)
    b = sum(r[q]["all20"] and not runs["0.7"][q]["all20"] for q in ids)
    c = sum(runs["0.7"][q]["all20"] and not r[q]["all20"] for q in ids)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    rows.append({"alpha": float(a), "R@1": round(m[1], 2), "R@5": round(m[5], 2), "R@20": round(m[20], 2),
                 "all@20": n_all, "b": b, "c": c, "p_vs_0.7": p})
    print(f"{a:<5}  {m[1]:5.1f}  {m[5]:5.1f}  {m[20]:5.1f}  {n_all:4d}    {b:3d}:{c:<3d}   {p:.3g}")
(D / "summary.json").write_text(json.dumps({"n": len(ids), "rows": rows}, ensure_ascii=False, indent=1))
