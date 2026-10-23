"""상한 진단 분석 — O1/O2/O3(run1) 대 D(882: 3,072 3회 다수결, 4,096 3회가 다 있으면 그것도). 층별 표 + 성분(O1−D, O3−D, O1−O2, O2−D) + 같은 문항 McNemar.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/ceiling_20261019/analyze.py
"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = official_answers("train"); V = "R1X1F1B2D1C1E0"; FD = ROOT / "results/format_dev_20261017"
LAYERS = ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")
def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return {r["query_id"]: sf.score(r, G[r["query_id"]]) for r in rows}
def majority(runs):
    qs = set.intersection(*(set(r) for r in runs))
    return {q: {"fin_inc": int(sum(r[q]["fin_inc"] for r in runs) * 2 > len(runs)), "layer": runs[0][q]["layer"], "trunc": sum(r[q]["trunc"] for r in runs) / len(runs), "n_out": sum(r[q]["n_out"] for r in runs) / len(runs)} for q in qs}
arms = {}
d3 = [load(FD / d / "D" / f"{V}.jsonl") for d in ("final882_cap3072_qwen3_5_4b", "final882_cap3072_run2", "final882_cap3072_run3")]
arms["D 3,072 다수결(3회)"] = majority(d3); arms["D 3,072 실행1"] = d3[0]
d4 = [FD / f"final882_cap4096_run{i}" / "D" / f"{V}.jsonl" for i in (1, 2, 3)]
if all(p.exists() for p in d4):
    r4 = [load(p) for p in d4]
    if all(len(r) == 882 for r in r4): arms["D 4,096 다수결(3회)"] = majority(r4)
for o in ("O1", "O2", "O3"):
    p = HERE / "run1" / o / f"{V}.jsonl"
    if p.exists():
        s = load(p)
        if len(s) == 882: arms[{"O1": "O1 정답 셀만", "O2": "O2 정답 표 전체", "O3": "O3 정답 표 안 D"}[o]] = s
        else: print(f"[{o}] {len(s)}/882 진행 중")
qs = sorted(set.intersection(*(set(s) for s in arms.values())))
print(f"문항 {len(qs)}\n| 입력 | 맞힘 | % | " + " | ".join(LAYERS) + " | 잘림 | 출력 토큰 |\n|---|---|---|" + "---|" * len(LAYERS) + "---|---|")
for name, s in arms.items():
    tot = sum(s[q]["fin_inc"] for q in qs)
    per = [f"{sum(s[q]['fin_inc'] for q in qs if s[q]['layer']==L)}/{sum(1 for q in qs if s[q]['layer']==L)}" for L in LAYERS]
    print(f"| {name} | {tot} | {100*tot/len(qs):.1f} | " + " | ".join(per) + f" | {sum(s[q]['trunc'] for q in qs):.0f} | {sum(s[q]['n_out'] for q in qs)/len(qs):.0f} |")
def mc(x, y):
    b = sum(arms[x][q]["fin_inc"] and not arms[y][q]["fin_inc"] for q in qs); c = sum(arms[y][q]["fin_inc"] and not arms[x][q]["fin_inc"] for q in qs)
    return f"{x} 대 {y}: {sum(arms[x][q]['fin_inc'] for q in qs)} 대 {sum(arms[y][q]['fin_inc'] for q in qs)} (차 {sum(arms[x][q]['fin_inc'] for q in qs)-sum(arms[y][q]['fin_inc'] for q in qs):+d}), b:c {b}:{c} p={binomtest(min(b,c), b+c, 0.5).pvalue if b+c else 1.0:.4f}"
Dm = "D 3,072 다수결(3회)"
print("\n== 성분 ==")
for pair in (("O1 정답 셀만", Dm), ("O3 정답 표 안 D", Dm), ("O2 정답 표 전체", Dm), ("O1 정답 셀만", "O2 정답 표 전체"), ("O1 정답 셀만", "O3 정답 표 안 D")):
    if pair[0] in arms and pair[1] in arms: print("  " + mc(*pair))
if "D 4,096 다수결(3회)" in arms:
    for pair in (("O1 정답 셀만", "D 4,096 다수결(3회)"), ("O3 정답 표 안 D", "D 4,096 다수결(3회)")):
        if pair[0] in arms: print("  " + mc(*pair))
if "O1 정답 셀만" in arms:
    o1w = [q for q in qs if not arms["O1 정답 셀만"][q]["fin_inc"]]
    print(f"\nO1 틀림 {len(o1w)}: 층별 " + ", ".join(f"{L} {sum(arms['O1 정답 셀만'][q]['layer']==L for q in o1w)}" for L in LAYERS) + f" | 그중 D 다수결도 틀림 {sum(not arms[Dm][q]['fin_inc'] for q in o1w)} | O1 틀림·D 맞음 {sum(arms[Dm][q]['fin_inc'] for q in o1w)}")
    json.dump(o1w, open(HERE / "O1_wrong_ids.json", "w"))
