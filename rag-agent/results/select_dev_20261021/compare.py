"""고르기 있음(AF·DF·CpF, run1·run2) 대 없음(sentence_dev run1·run2 의 A·D·Cp) — PREREG-2026-10-21-select-dev.md. 같은 실행 번호끼리 짝, 실행별 McNemar, 2회 평균.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/select_dev_20261021/compare.py
"""
import json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts")]
import score_formats2 as sf
from format_render import load_split
V = "R1X1F1B2D1C1E0"; SD = ROOT / "results/sentence_dev_20261019"
G = {r["query_id"]: r["answer"] for r in map(json.loads, open(ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl", encoding="utf-8"))}
def load(p): return {r["query_id"]: sf.score(r, G[r["query_id"]])["fin_inc"] for r in map(json.loads, open(p, encoding="utf-8"))}
pairs = [("A", "AF", "LangChain 청크"), ("D", "DF", "잘라낸 표"), ("Cp", "CpF", "셀 문장 목록")]
B = {a: [load(SD / f"run{i}" / a / f"{V}.jsonl") for i in (1, 2)] for a, _, _ in pairs}
N_ = {f: [load(HERE / f"run{i}" / f / f"{V}.jsonl") for i in (1, 2)] for _, f, _ in pairs}
qs = sorted(B["A"][0]); N = len(qs)
def mcn(x, y):
    b = sum(x[q] and not y[q] for q in qs); c = sum(y[q] and not x[q] for q in qs)
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)
F = {f: {r["query_id"]: r for r in map(json.loads, open(HERE / f"{f}_records.jsonl", encoding="utf-8"))} for _, f, _ in pairs}
print(f"문항 {N}\n== 고르기 있음 대 없음(입력마다, 실행별; 마지막 줄 맞힘) ==\n| 입력 | 실행 | 없음 | 있음 | 차이 | 있음만:없음만 | p |\n|---|---|---|---|---|---|---|")
for a, f, nm in pairs:
    for i in range(2):
        x = sum(N_[f][i].values()); y = sum(B[a][i].values()); b, c, p = mcn(N_[f][i], B[a][i])
        print(f"| {nm} | {i+1} | {y} | {x} | {x-y:+d} | {b}:{c} | {p:.3f} |")
    mx = sum(sum(S.values()) for S in N_[f]) / 2; my = sum(sum(S.values()) for S in B[a]) / 2
    print(f"| {nm} | 평균 | {my:.1f} | {mx:.1f} | {mx-my:+.1f} | | |")
print("\n== 고르기 있음 아래 입력 간(실행별) ==")
for i in range(2):
    for x, y in (("DF", "AF"), ("DF", "CpF"), ("CpF", "AF")):
        b, c, p = mcn(N_[x][i], N_[y][i]); print(f"  실행{i+1} {x} 대 {y}: {sum(N_[x][i].values())} 대 {sum(N_[y][i].values())}, b:c {b}:{c}, p={p:.4f}")
print("\n== 고르기 통계 ==")
tables, hdr, sent, coord = load_split("validation", qs)
for a, f, nm in pairs:
    R = F[f]; n = len(R)
    s = f"  {nm}: 줄 {sum(r['n_lines'] for r in R.values())/n:.1f} → 남김 {sum(r['n_kept'] for r in R.values())/n:.1f}, 토큰 {sum(r['n_tok_ctx_before'] for r in R.values())/n:.0f} → {sum(r['n_tok_ctx'] for r in R.values())/n:.0f}, 빈 문맥 {sum(r['empty'] for r in R.values())}, 선택 출력 잘림 {sum(not r['sel_finished'] for r in R.values())}"
    if a == "Cp":
        keep_all = sum(all(sent[q][g] in R[q]["context"] for g in R[q]["gold_ids"] if g in sent[q]) for q in qs)
        orig = {r["query_id"]: r for r in map(json.loads, open(SD / "Cp_records.jsonl", encoding="utf-8"))}
        had_all = sum(all(sent[q][g] in orig[q]["context"] for g in R[q]["gold_ids"] if g in sent[q]) for q in qs)
        s += f", 정답 셀 줄 전부 있음: 고르기 전 {had_all} → 후 {keep_all}"
    print(s)
json.dump({"n": N, "base": {a: [sum(S.values()) for S in B[a]] for a in B}, "new": {f: [sum(S.values()) for S in N_[f]] for f in N_}}, open(HERE / "compare.json", "w"), indent=1)
