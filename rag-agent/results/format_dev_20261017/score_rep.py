"""반복 실행 판정 — PREREG-2026-10-18-D-vs-A-replication.md. 문항마다 3회 중 2회 이상 맞힘(다수결) → 같은 문항 정확 McNemar.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_rep.py <run-dir> <run-dir> <run-dir> [--gold train]
"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
dirs = [a for a in sys.argv[1:] if not a.startswith("--") and a != "train"]
G = official_answers("train") if "--gold" in sys.argv else {r["query_id"]: r["answer"] for r in map(json.loads, open(ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl", encoding="utf-8"))}
V = "R1X1F1B2D1C1E0"
runs = []
for d in dirs:
    S = {}
    for arm in ("D", "C", "A"):
        p = Path(d) / arm / f"{V}.jsonl"
        S[arm] = {r["query_id"]: sf.score(r, G[r["query_id"]]) for r in map(json.loads, open(p, encoding="utf-8"))}
    runs.append(S)
qs = sorted(set.intersection(*[set(S[a]) for S in runs for a in S])); N = len(qs)
maj = {a: {q: int(sum(S[a][q]["fin_inc"] for S in runs) >= 2) for q in qs} for a in ("D", "C", "A")}
all3 = {a: sum(all(S[a][q]["fin_inc"] for S in runs) for q in qs) for a in ("D", "C", "A")}
def mcn(x, y):
    b = sum(x[q] and not y[q] for q in qs); c = sum(y[q] and not x[q] for q in qs)
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)
print(f"문항 {N}, 실행 {len(runs)}회")
print("| 팔 | " + " | ".join(f"실행{i+1}" for i in range(len(runs))) + " | 평균 | 다수결 맞힘 | 3회 전부 맞힘 |")
print("|---|" + "---|" * (len(runs) + 3))
for a in ("D", "C", "A"):
    vals = [sum(S[a][q]["fin_inc"] for q in qs) for S in runs]
    print(f"| {a} | " + " | ".join(str(v) for v in vals) + f" | {sum(vals)/len(vals):.1f} | {sum(maj[a].values())} ({100*sum(maj[a].values())/N:.1f}%) | {all3[a]} |")
print("\n다수결 쌍 비교(정확 McNemar):")
for x, y in (("D", "A"), ("D", "C"), ("C", "A")):
    b, c, p = mcn(maj[x], maj[y]); print(f"  {x} 대 {y}: {sum(maj[x].values())} 대 {sum(maj[y].values())}, b:c {b}:{c}, p={p:.4f}")
print("\n실행별 쌍 비교(마지막 줄):")
for i, S in enumerate(runs, 1):
    for x, y in (("D", "A"), ("D", "C"), ("C", "A")):
        b = sum(S[x][q]["fin_inc"] and not S[y][q]["fin_inc"] for q in qs); c = sum(S[y][q]["fin_inc"] and not S[x][q]["fin_inc"] for q in qs)
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        print(f"  실행{i} {x} 대 {y}: {sum(S[x][q]['fin_inc'] for q in qs)} 대 {sum(S[y][q]['fin_inc'] for q in qs)}, b:c {b}:{c}, p={p:.4f}")
print("\n실행 간 같은 팔 차이(마지막 줄 맞힘 불일치 b:c):")
for a in ("D", "C", "A"):
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            b = sum(runs[i][a][q]["fin_inc"] and not runs[j][a][q]["fin_inc"] for q in qs); c = sum(runs[j][a][q]["fin_inc"] and not runs[i][a][q]["fin_inc"] for q in qs)
            print(f"  {a}: 실행{i+1} 대 실행{j+1} {b}:{c}")
# 층별 다수결
layers = {}
for q in qs: layers.setdefault(runs[0]["D"][q]["layer"], []).append(q)
print("\n층별 다수결 맞힘 (D / C / A):")
for L, lq in sorted(layers.items()):
    print(f"  {L} ({len(lq)}): {sum(maj['D'][q] for q in lq)} / {sum(maj['C'][q] for q in lq)} / {sum(maj['A'][q] for q in lq)}")
json.dump({"n": N, "majority": {a: sum(v.values()) for a, v in maj.items()}, "all3": all3,
           "pairs_majority": {f"{x}_vs_{y}": dict(zip(("b", "c", "p"), mcn(maj[x], maj[y]))) for x, y in (("D", "A"), ("D", "C"), ("C", "A"))}},
          open(HERE / "score_882_rep_majority.json", "w"), indent=1)
