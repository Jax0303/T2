"""self-consistency(5표본 다수결) 대 greedy(sentence_dev run1·run2), dev 395, 입력별. PREREG-2026-10-22-self-consistency-dev.md"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
V = "R1X1F1B2D1C1E0"; SD = ROOT / "results/sentence_dev_20261019"
G = {r["query_id"]: r["answer"] for r in map(json.loads, open(ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl", encoding="utf-8"))}
name = {"D": "잘라낸 표", "Cp": "셀 문장 목록", "A": "LangChain 청크"}
def load(p): return {r["query_id"]: (sf.score(r, G[r["query_id"]])["fin_inc"], r) for r in map(json.loads, open(p, encoding="utf-8"))}
def mcn(x, y, qs):
    b = sum(x[q] and not y[q] for q in qs); c = sum(y[q] and not x[q] for q in qs); return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)
S = {}; Gr = {}
for a in ("D", "Cp", "A"):
    p = HERE / "run1" / a / f"{V}.jsonl"
    if not p.exists(): continue
    S[a] = load(p); Gr[a] = [load(SD / f"run{i}" / a / f"{V}.jsonl") for i in (1, 2)]
    qs = sorted(S[a]); sc = {q: S[a][q][0] for q in qs}; rows = S[a]
    g = [{q: R[q][0] for q in qs} for R in Gr[a]]
    agree = sum(rows[q][1].get("sc_agree", 0) for q in qs); votes = sum(rows[q][1].get("sc_votes", 0) for q in qs) / len(qs)
    tok = sum(rows[q][1].get("sc_n_tok_out_all", 0) for q in qs) / len(qs); tok1 = sum(sum(R[q][1]["n_tok_out"] for q in qs) for R in Gr[a]) / 2 / len(qs)
    print(f"{name[a]}: greedy {[sum(x.values()) for x in g]} 평균 {sum(sum(x.values()) for x in g)/2:.1f} → 다수결 5표본 {sum(sc.values())} ({sum(sc.values())-sum(sum(x.values()) for x in g)/2:+.1f}); "
          + "; ".join(f"대 greedy{i+1} b:c {b}:{c} p={p:.3f}" for i, (b, c, p) in enumerate(mcn(sc, x, qs) for x in g))
          + f" | 5표본 전원 일치 {agree}/{len(qs)}, 평균 득표 {votes:.2f}, 출력 토큰 문항당 {tok:.0f} (greedy {tok1:.0f})")
qs = sorted(set.intersection(*[set(S[a]) for a in S])) if S else []
for x, y in (("D", "A"), ("D", "Cp"), ("Cp", "A")):
    if x in S and y in S:
        b, c, p = mcn({q: S[x][q][0] for q in qs}, {q: S[y][q][0] for q in qs}, qs); print(f"  다수결 아래 {name[x]} 대 {name[y]}: {sum(S[x][q][0] for q in qs)} 대 {sum(S[y][q][0] for q in qs)} b:c {b}:{c} p={p:.3f}")
