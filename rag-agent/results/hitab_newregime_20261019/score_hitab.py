"""HiTab 새 설정 채점: 마지막 'Final answer' 줄(굵은 라벨 허용)에 정답 원소가 모두 포함(형식 허용 v2)되면 맞힘; 보조 = 공식 hitab_exact_match_text(마지막 줄).
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/hitab_newregime_20261019/score_hitab.py <run-dir> [<run-dir> ...]
"""
import ast, json, os, re, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
import score_formats2 as sf
from score_v2 import included_v2
from rag_agent.eval.metrics import hitab_exact_match_text
V = "R0X1F1B2D1C1E0"
BASE = Path(os.environ.get("HITAB_DIR", HERE))  # 확인 표본은 HITAB_DIR=.../confirm
A = {r["query_id"]: r for r in map(json.loads, open(BASE / "A_records.jsonl", encoding="utf-8"))}
N = len(A)
def gold_list(ans):
    if isinstance(ans, list): return [str(x) for x in ans]
    try:
        v = ast.literal_eval(str(ans)); return [str(x) for x in v] if isinstance(v, list) else [str(v)]
    except Exception: return [str(ans)]
def norm_fl(fl):  # 2026-10-19 결과 뒤 추가(모든 팔 동일): 앞에 0 없는 소수(.544 → 0.544), 부동소수 표기 잡음(-0.20800000000000002 → -0.208). MultiHiertt 882 에는 영향 0(score_fix_check.py).
    if fl is None: return None
    fl = re.sub(r"(?<![\d.])\.(\d+)", r"0.\1", fl)
    return re.sub(r"-?\d+\.\d{7,}", lambda m: format(float(m.group(0)), ".10g"), fl)
def score(r):
    g = gold_list(A[r["query_id"]]["answer"]); fl = norm_fl(sf.final_line(r["raw"]))
    fin = int(fl is not None and all(included_v2(fl, x, False) for x in g))
    try: off = int(fl is not None and hitab_exact_match_text(fl, ast.literal_eval(str(A[r["query_id"]]["answer"])) if str(A[r["query_id"]]["answer"]).startswith("[") else A[r["query_id"]]["answer"]))
    except Exception: off = 0
    return {"fin_inc": fin, "official": off, "trunc": int(not r["finished"]), "n_out": r["n_tok_out"]}
runs = {}
for d in sys.argv[1:]:
    S = {}
    for arm in ("A", "C", "D", "Cp"):
        p = Path(d) / arm / f"{V}.jsonl"
        if p.exists(): S[arm] = {r["query_id"]: score(r) for r in map(json.loads, open(p, encoding="utf-8"))}
    runs[Path(d).name] = S
qs = sorted(A)
for rn, S in runs.items():
    print(f"\n== {rn} ==\n| 팔 | 마지막 줄 맞힘 /{N} | % | 공식 EM | 잘림 | 출력 토큰 |\n|---|---|---|---|---|---|")
    for arm, s in S.items():
        if len(s) < N: print(f"| {arm} | {len(s)}/{N} 진행 중 |"); continue
        print(f"| {arm} | {sum(s[q]['fin_inc'] for q in qs)} | {100*sum(s[q]['fin_inc'] for q in qs)/N:.1f} | {sum(s[q]['official'] for q in qs)/N:.3f} | {sum(s[q]['trunc'] for q in qs)} | {sum(s[q]['n_out'] for q in qs)/N:.0f} |")
    for x, y in (("D", "A"), ("D", "C"), ("C", "A"), ("Cp", "C"), ("Cp", "A"), ("D", "Cp")):
        if x in S and y in S and len(S[x]) == len(S[y]) == N:
            b = sum(S[x][q]["fin_inc"] and not S[y][q]["fin_inc"] for q in qs); c = sum(S[y][q]["fin_inc"] and not S[x][q]["fin_inc"] for q in qs)
            print(f"  {x} 대 {y}: b:c {b}:{c} p={binomtest(min(b,c), b+c, 0.5).pvalue if b+c else 1.0:.3f}")
if len(runs) > 1:
    print("\n== 실행 평균 ==")
    for arm in ("A", "C", "D", "Cp"):
        vals = [sum(S[arm][q]["fin_inc"] for q in qs) for S in runs.values() if arm in S and len(S[arm]) == N]
        if vals: print(f"  {arm}: {vals} 평균 {sum(vals)/len(vals):.1f}")
if len(runs) >= 3:
    print("\n== 다수결(3회) ==")
    maj = {}
    for arm in ("A", "C", "D", "Cp"):
        Ss = [S[arm] for S in runs.values() if arm in S and len(S[arm]) == N]
        if len(Ss) >= 3: maj[arm] = {q: int(sum(s[q]["fin_inc"] for s in Ss) * 2 > len(Ss)) for q in qs}
    for arm, m in maj.items(): print(f"  {arm}: 다수결 맞힘 {sum(m.values())} ({100*sum(m.values())/N:.1f}%)")
    for x, y in (("D", "A"), ("D", "C"), ("C", "A"), ("Cp", "C"), ("Cp", "A"), ("D", "Cp")):
        if x in maj and y in maj:
            b = sum(maj[x][q] and not maj[y][q] for q in qs); c = sum(maj[y][q] and not maj[x][q] for q in qs)
            print(f"  {x} 대 {y}(다수결): {sum(maj[x].values())} 대 {sum(maj[y].values())}, b:c {b}:{c} p={binomtest(min(b,c), b+c, 0.5).pvalue if b+c else 1.0:.4f}")
