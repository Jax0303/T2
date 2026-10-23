"""dev 332 후보 채점 — PREREG-2026-10-15-prf.md. 선택 지표 = 정답 포함 v2, 예/아니오 정답 문항 제외. 기준 C = cellform_dev ans_C.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/prf_20261015/score_dev.py
"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
from mh_arms import official_answers                       # noqa: E402
from score_inclusion import gold_kind                      # noqa: E402
from score_v2 import included_v2                           # noqa: E402

D = ROOT / "results/prf_20261015/dev"
G = official_answers("validation")
ids = [json.loads(l)["query_id"] for l in open(ROOT / "results/reader_format_20260927/mh_dev_pop332.jsonl") if l.strip()]
sel = [q for q in ids if gold_kind(G[q]) != "yesno"]
load = lambda d: {r["query_id"]: r for r in map(json.loads, open(d / "langchain_chunk_answer.jsonl", encoding="utf-8"))}
cands = {"C(기준)": ROOT / "results/cellform_dev_20261012/ans_C"}
for c in ("prf1_path", "prf3_path", "prf5_path", "prf1_full", "prf3_full", "prf5_full"):
    if (D / f"ans_{c}" / "langchain_chunk_answer.jsonl").exists():
        cands[c] = D / f"ans_{c}"
S = {}
print(f"dev 332 중 예/아니오 정답 {len(ids) - len(sel)} 제외 → 선택 지표 문항 {len(sel)}")
print("| 후보 | 답변 완료 | 정답 포함(검색) | 맞힘(선택 지표) | 기준 대비 b:c p |")
print("|---|---|---|---|---|")
base = None
for name, d in cands.items():
    A = load(d)
    if len(A) < len(ids):
        print(f"| {name} | {len(A)}/332 (진행 중) | | | |"); continue
    S[name] = {q: int(included_v2(A[q]["raw"], G[q], A[q]["kind"] == "arith")) for q in sel}
    cov = sum(A[q]["retrieval_correct"] for q in ids)
    n = sum(S[name].values())
    if base is None:
        base = S[name]; print(f"| {name} | 332 | {cov} | {n} ({100*n/len(sel):.1f}%) | — |")
    else:
        bb = sum(base[q] and not S[name][q] for q in sel); cc = sum(S[name][q] and not base[q] for q in sel)
        p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if bb + cc else 1.0
        print(f"| {name} | 332 | {cov} | {n} ({100*n/len(sel):.1f}%) | {n - sum(base.values()):+d} ({bb}:{cc} p={p:.3f}) |")
