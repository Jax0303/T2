"""표 2개+ 새 문항 — P* 세 입력 3회: 잘라낸 표 대 LangChain(주), 대 목록, 목록 대 LangChain, 둘 다 정답 포함 문항 격차. PREREG-2026-10-22-multitable-confirm.md"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = official_answers("train"); V = "R1X1F1B2D1C1E0"; name = {"D": "잘라낸 표", "C": "셀 문장 목록", "A": "LangChain 청크"}
def load(p): return {r["query_id"]: (sf.score(r, G[r["query_id"]])["fin_inc"], int(bool(r["retrieval_correct"]))) for r in map(json.loads, open(p, encoding="utf-8"))}
mc = lambda b, c: binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
S = {a: [load(HERE / f"run{i}" / a / f"{V}.jsonl") for i in (1, 2, 3)] for a in "DCA"}
qs = sorted(set.intersection(*[set(x) for a in S for x in S[a]])); N = len(qs)
print(f"== 표 2개+ 새 문항 {N}, P*, 마지막 줄 포함 채점 ==")
for a in "DCA": print(f"  {name[a]}: {[sum(x[q][0] for q in qs) for x in S[a]]} 평균 {sum(sum(x[q][0] for q in qs) for x in S[a])/3:.1f} ({100*sum(sum(x[q][0] for q in qs) for x in S[a])/3/N:.1f}%) | 정답 포함 {sum(S[a][0][q][1] for q in qs)}")
for x, y in (("D", "A"), ("D", "C"), ("C", "A")):
    rows = []
    for i in range(3):
        b = sum(S[x][i][q][0] and not S[y][i][q][0] for q in qs); c = sum(S[y][i][q][0] and not S[x][i][q][0] for q in qs)
        rows.append((sum(S[x][i][q][0] for q in qs) - sum(S[y][i][q][0] for q in qs), b, c, mc(b, c)))
    print(f"  {name[x]} 대 {name[y]}: " + "; ".join(f"실행{i+1} {d:+d} ({b}:{c}, p={p:.3f})" for i, (d, b, c, p) in enumerate(rows)) + f" | 3회 평균 {sum(r[0] for r in rows)/3/N*100:+.1f}%p | 세 실행 모두 {name[x]}>: {all(r[0]>0 for r in rows)}, 2회 이상 p<.05: {sum(r[3]<.05 for r in rows)>=2}")
both = [q for q in qs if S["D"][0][q][1] and S["A"][0][q][1]]
print(f"  둘 다 정답 포함 {len(both)}문항 잘라낸 표−LangChain: " + "; ".join(f"실행{i+1} {sum(S['D'][i][q][0] for q in both)-sum(S['A'][i][q][0] for q in both):+d}" for i in range(3)))
