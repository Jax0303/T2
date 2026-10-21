"""지시문 4개 × 세 입력 × 3회(600b) — 지시문마다 D 대 A, D 대 C, C 대 A 실행별 McNemar와 3회 평균. PREREG-2026-10-22-multiprompt-600b.md
채점: 'Final answer' 줄이 없는 지시문(C0)은 전체 포함(inc), 있는 지시문은 마지막 줄 포함(fin_inc)."""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = official_answers("train")
VS = [("R0X0F0B0D0C0E0", "원문 LangChain", "inc"), ("R0X0F0B2D0C1E0", "원문+최소 변경", "fin_inc"), ("R1X1F1B2D1C1E0", "P*", "fin_inc"), ("R1X3F1B2D1C1E0", "P*+꺽쇠 설명", "fin_inc")]
name = {"D": "잘라낸 표", "C": "셀 문장 목록", "A": "LangChain 청크"}
def load(p, key): return {r["query_id"]: sf.score(r, G[r["query_id"]])[key] for r in map(json.loads, open(p, encoding="utf-8"))}
out = {}
for v, vn, key in VS:
    S = {a: [load(HERE / f"run{i}" / a / f"{v}.jsonl", key) for i in (1, 2, 3)] for a in "DCA"}
    qs = sorted(set.intersection(*[set(x) for a in S for x in S[a]])); N = len(qs)
    print(f"\n== {vn} ({v}, 채점 {key}, 문항 {N}) ==")
    for a in "DCA": print(f"  {name[a]}: {[sum(x[q] for q in qs) for x in S[a]]} 평균 {sum(sum(x[q] for q in qs) for x in S[a])/3:.1f}")
    for x, y in (("D", "A"), ("D", "C"), ("C", "A")):
        rows = []
        for i in range(3):
            b = sum(S[x][i][q] and not S[y][i][q] for q in qs); c = sum(S[y][i][q] and not S[x][i][q] for q in qs)
            rows.append((sum(S[x][i][q] for q in qs) - sum(S[y][i][q] for q in qs), b, c, binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0))
        print(f"  {name[x]} 대 {name[y]}: " + "; ".join(f"실행{i+1} {d:+d} ({b}:{c}, p={p:.3f})" for i, (d, b, c, p) in enumerate(rows)) + f" | 세 실행 모두 {name[x]}>: {all(r[0]>0 for r in rows)}, 2회 이상 p<.05: {sum(r[3]<.05 for r in rows)>=2}")
    out[v] = {a: [sum(x[q] for q in qs) for x in S[a]] for a in "DCA"}
json.dump(out, open(HERE / "compare.json", "w"), indent=1)
