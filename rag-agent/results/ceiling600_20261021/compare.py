"""O1·O3(1회) 대 D·C·A(기존 run1~3, 같은 600) 비교 — PREREG-2026-10-21-ceiling-600.md. 판정 = O 1회 대 기존 실행별 McNemar(3개), 3회 평균 차이.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/ceiling600_20261021/compare.py
"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = official_answers("train"); V = "R1X1F1B2D1C1E0"; MC = ROOT / "results/mh_confirm_20261019"
def load(p): return {r["query_id"]: sf.score(r, G[r["query_id"]])["fin_inc"] for r in map(json.loads, open(p, encoding="utf-8"))}
O = {a: load(HERE / "run1" / a / f"{V}.jsonl") for a in ("O1", "O3")}
B = {a: [load(MC / f"run{i}" / a / f"{V}.jsonl") for i in (1, 2, 3)] for a in ("D", "C", "A")}
qs = sorted(O["O1"]); N = len(qs)
def mcn(x, y):
    b = sum(x[q] and not y[q] for q in qs); c = sum(y[q] and not x[q] for q in qs)
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)
name = {"O1": "정답 셀만", "O3": "정답 표 안 잘라낸 표", "D": "잘라낸 표", "C": "셀 문장 목록", "A": "LangChain 청크"}
print(f"문항 {N}\n| 조건 | 맞힘 | % |\n|---|---|---|")
for a in ("O1", "O3"): print(f"| {name[a]} (1회) | {sum(O[a].values())} | {100*sum(O[a].values())/N:.1f} |")
for a in ("D", "C", "A"):
    s = [sum(S.values()) for S in B[a]]; print(f"| {name[a]} (3회 {s}) | {sum(s)/3:.1f} | {100*sum(s)/3/N:.1f} |")
print("\n== 상한 대 기존(실행별 McNemar) ==")
for o in ("O1", "O3"):
    for a in ("D", "C", "A"):
        rows = [mcn(O[o], S) for S in B[a]]
        d = sum(O[o].values()) - sum(sum(S.values()) for S in B[a]) / 3
        print(f"  {name[o]} 대 {name[a]}: 3회 평균 차이 {d:+.1f}; " + "; ".join(f"실행{i+1} b:c {b}:{c} p={p:.4f}" for i, (b, c, p) in enumerate(rows)))
b, c, p = mcn(O["O1"], O["O3"]); print(f"  정답 셀만 대 정답 표 안: {sum(O['O1'].values())} 대 {sum(O['O3'].values())}, b:c {b}:{c}, p={p:.4f}")
allD = {q for q in qs if all(S[q] for S in B["D"])}; noneD = {q for q in qs if not any(S[q] for S in B["D"])}
print(f"\nD 3회 다 틀림 {len(noneD)} 중 정답 셀만 맞힘 {sum(O['O1'][q] for q in noneD)}, 정답 표 안 맞힘 {sum(O['O3'][q] for q in noneD)} | D 3회 다 맞힘 {len(allD)} 중 정답 셀만 틀림 {sum(not O['O1'][q] for q in allD)}")
json.dump({"n": N, "O1": sum(O["O1"].values()), "O3": sum(O["O3"].values()), "base": {a: [sum(S.values()) for S in B[a]] for a in B}}, open(HERE / "compare.json", "w"), indent=1)
