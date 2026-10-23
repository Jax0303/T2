"""D 대 A 차이가 정답 셀 포함 차이로 설명되는지: 둘 다 포함 / D만 / A만 / 둘 다 미포함으로 갈라 세기. 기존 결과 파일만 읽음."""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
ROOT = Path("/home/ugh/T2/rag-agent")
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = official_answers("train")
jobs = json.load(open(ROOT / "results/budget_forcing_20261022/jobs.json"))
P = "R1X1F1B2D1C1E0"
def load(p):
    d = {}
    for r in map(json.loads, open(p, encoding="utf-8")):
        d[r["query_id"]] = (sf.score(r, G[r["query_id"]])["fin_inc"], int(bool(r["retrieval_correct"])))
    return d
for S in ("600", "600b", "882"):
    runs = sorted({j["run"] for j in jobs if j["set"] == S})
    print(f"\n=== {S} (P*) ===")
    agg = {}
    for run in runs:
        J = {j["arm"]: j["out"] for j in jobs if j["set"] == S and j["run"] == run and j["variant"] == P}
        D, A, C = load(J["D"]), load(J["A"]), load(J["C"])
        qs = sorted(set(D) & set(A) & set(C))
        gi = {a: sum(x[q][1] for q in qs) for a, x in (("D", D), ("A", A), ("C", C))}
        acc = {a: sum(x[q][0] for q in qs) for a, x in (("D", D), ("A", A), ("C", C))}
        # 정답 포함 여부로 가르기 (D 대 A)
        cells = {}
        for q in qs:
            key = (D[q][1], A[q][1])
            c = cells.setdefault(key, [0, 0, 0, 0, 0])  # n, D맞힘, A맞힘, D만 맞힘, A만 맞힘
            c[0] += 1; c[1] += D[q][0]; c[2] += A[q][0]; c[3] += D[q][0] and not A[q][0]; c[4] += A[q][0] and not D[q][0]
        both = cells.get((1, 1), [0]*5)
        p = binomtest(min(both[3], both[4]), both[3] + both[4], 0.5).pvalue if both[3] + both[4] else 1.0
        print(f"{run}: 정답 포함 D {gi['D']} C {gi['C']} A {gi['A']} | 맞힘 D {acc['D']} C {acc['C']} A {acc['A']} | 둘 다 포함 {both[0]}문항: D {both[1]} 대 A {both[2]} ({both[3]}:{both[4]}, p={p:.3f}) | D만 포함 {cells.get((1,0),[0]*5)[0]}문항: D {cells.get((1,0),[0]*5)[1]} A {cells.get((1,0),[0]*5)[2]} | A만 포함 {cells.get((0,1),[0]*5)[0]}문항: D {cells.get((0,1),[0]*5)[1]} A {cells.get((0,1),[0]*5)[2]} | 둘 다 미포함 {cells.get((0,0),[0]*5)[0]}문항: D {cells.get((0,0),[0]*5)[1]} A {cells.get((0,0),[0]*5)[2]}")
        # C 대 A 둘 다 포함
        cb = [0, 0, 0, 0, 0]
        for q in qs:
            if C[q][1] and A[q][1]:
                cb[0] += 1; cb[1] += C[q][0]; cb[2] += A[q][0]; cb[3] += C[q][0] and not A[q][0]; cb[4] += A[q][0] and not C[q][0]
        pc = binomtest(min(cb[3], cb[4]), cb[3] + cb[4], 0.5).pvalue if cb[3] + cb[4] else 1.0
        print(f"      C 대 A 둘 다 포함 {cb[0]}문항: C {cb[1]} 대 A {cb[2]} ({cb[3]}:{cb[4]}, p={pc:.3f})")
