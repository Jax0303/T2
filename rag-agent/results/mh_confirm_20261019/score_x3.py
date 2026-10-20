"""꺽쇠 설명 지시문 있음(new, R1X3…) 대 없음(base, R1X1…) 짝 비교 — PREREG-2026-10-20-bracket-explain.md.
같은 600 문항, 실행 번호끼리 짝(run1↔run1x3 …). 보고 = 실행별 맞힘 + 3회 평균(다수결 없음), 판정 = 실행별 정확 McNemar.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/mh_confirm_20261019/score_x3.py --base run1 run2 run3 --new run1x3 run2x3 run3x3
"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = official_answers("train")
VB, VN = "R1X1F1B2D1C1E0", "R1X3F1B2D1C1E0"
args = sys.argv[1:]; base = args[args.index("--base") + 1:args.index("--new")]; new = args[args.index("--new") + 1:]
def load(d, v):
    return {a: {r["query_id"]: sf.score(r, G[r["query_id"]])["fin_inc"] for r in map(json.loads, open(Path(d) / a / f"{v}.jsonl", encoding="utf-8"))} for a in ("D", "C", "A")}
B = [load(d, VB) for d in base]; Nw = [load(d, VN) for d in new]
qs = sorted(set.intersection(*[set(S[a]) for S in B + Nw for a in S])); N = len(qs)
def mcn(x, y):
    b = sum(x[q] and not y[q] for q in qs); c = sum(y[q] and not x[q] for q in qs)
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)
name = {"D": "잘라낸 표", "C": "셀 문장 목록", "A": "LangChain 청크"}
print(f"문항 {N}, 실행 {len(B)}회 짝 비교(같은 실행 번호끼리)\n")
print("== 꺽쇠 설명 있음 대 없음(입력마다, 실행별) ==")
print("| 입력 | 실행 | 없음 | 있음 | 차이 | 있음만 맞힘:없음만 맞힘 | p |\n|---|---|---|---|---|---|---|")
for a in ("D", "C", "A"):
    for i in range(len(B)):
        x = sum(Nw[i][a][q] for q in qs); y = sum(B[i][a][q] for q in qs); b, c, p = mcn(Nw[i][a], B[i][a])
        print(f"| {name[a]} | {i+1} | {y} | {x} | {x-y:+d} | {b}:{c} | {p:.3f} |")
    mx = sum(sum(S[a][q] for q in qs) for S in Nw) / len(Nw); my = sum(sum(S[a][q] for q in qs) for S in B) / len(B)
    print(f"| {name[a]} | 평균 | {my:.1f} | {mx:.1f} | {mx-my:+.1f} | | |")
print("\n== 꺽쇠 설명 아래 입력 간 비교(실행별) ==")
for i, S in enumerate(Nw, 1):
    for x, y in (("D", "A"), ("D", "C"), ("C", "A")):
        b, c, p = mcn(S[x], S[y])
        print(f"  실행{i} {name[x]} 대 {name[y]}: {sum(S[x][q] for q in qs)} 대 {sum(S[y][q] for q in qs)}, b:c {b}:{c}, p={p:.4f}")
json.dump({"n": N, "base": [{a: sum(S[a][q] for q in qs) for a in S} for S in B], "new": [{a: sum(S[a][q] for q in qs) for a in S} for S in Nw]}, open(HERE / "score_x3_pair.json", "w"), indent=1)
