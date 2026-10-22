"""정답 표 2개+ 문항에서 잘라낸 표 우세가 정답 포함(청크가 표를 놓침) 때문인지 읽기 때문인지 — 기존 기록만."""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
ROOT = Path("/home/ugh/T2/rag-agent")
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from mh_arms import official_answers
G = {**official_answers("train"), **official_answers("validation")}
jobs = json.load(open(ROOT / "results/budget_forcing_20261022/jobs.json")); P = "R1X1F1B2D1C1E0"
def load(p): return {r["query_id"]: (sf.score(r, G[r["query_id"]])["fin_inc"], int(bool(r["retrieval_correct"]))) for r in map(json.loads, open(p, encoding="utf-8"))}
def mc(b, c): return binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
for S in ("600", "600b", "882", "dev"):
    Arec = [j for j in jobs if j["set"] == S and j["variant"] == P and j["arm"] == "A"][0]["records"]
    ntab = {r["query_id"]: len({g.split("-")[0] for g in r["gold_ids"]}) for r in map(json.loads, open(Arec, encoding="utf-8"))}
    for run in sorted({j["run"] for j in jobs if j["set"] == S}):
        J = {j["arm"]: j["out"] for j in jobs if j["set"] == S and j["run"] == run and j["variant"] == P}
        D, A = load(J["D"]), load(J["A"])
        for grp, qs in (("표 1", [q for q in D if q in A and ntab[q] == 1]), ("표 2+", [q for q in D if q in A and ntab[q] >= 2])):
            gi = (sum(D[q][1] for q in qs), sum(A[q][1] for q in qs))
            both = [q for q in qs if D[q][1] and A[q][1]]
            b = sum(D[q][0] and not A[q][0] for q in both); c = sum(A[q][0] and not D[q][0] for q in both)
            donly = [q for q in qs if D[q][1] and not A[q][1]]
            print(f"{S} {run} {grp} {len(qs)}문항: 정답 포함 잘라낸 표 {gi[0]} / LangChain {gi[1]} | 둘 다 포함 {len(both)}: 잘라낸 표−LC {sum(D[q][0] for q in both)-sum(A[q][0] for q in both):+d} ({b}:{c}, p={mc(b,c):.2f}) | 잘라낸 표만 포함 {len(donly)} 중 잘라낸 표 맞힘 {sum(D[q][0] for q in donly)}, LC 맞힘 {sum(A[q][0] for q in donly)}")
