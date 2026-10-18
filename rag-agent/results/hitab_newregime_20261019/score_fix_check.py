"""채점 허용 범위 점검(결과 뒤 발견, 모든 팔에 동일 적용): (1) 앞에 0 없는 소수(.544)를 0.544 로, (2) 부동소수 표기 잡음(-0.20800000000000002)을 10 유효숫자로 정규화한 뒤 포함 채점.
HiTab 3회·882 3회(3,072)에 적용했을 때 팔별 맞힘 변화를 출력한다.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/hitab_newregime_20261019/score_fix_check.py
"""
import ast, json, re, sys
from pathlib import Path
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "results/format_dev_20261017"), str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
import score_formats2 as sf
from score_v2 import included_v2
from mh_arms import official_answers
def norm_fl(fl):
    if fl is None: return None
    fl = re.sub(r"(?<![\d.])\.(\d+)", r"0.\1", fl)                       # .544 → 0.544
    return re.sub(r"-?\d+\.\d{7,}", lambda m: format(float(m.group(0)), ".10g"), fl)  # -0.20800000000000002 → -0.208
H = ROOT / "results/hitab_newregime_20261019"; VH = "R0X1F1B2D1C1E0"
A = {r["query_id"]: r for r in map(json.loads, open(H / "A_records.jsonl", encoding="utf-8"))}
gl = lambda ans: [str(x) for x in ans] if isinstance(ans, list) else [str(ans)]
def hit_h(r, fix):
    fl = sf.final_line(r["raw"]); fl = norm_fl(fl) if fix else fl
    return int(fl is not None and all(included_v2(fl, x, False) for x in gl(A[r["query_id"]]["answer"])))
qs = sorted(A)
print("== HiTab 300 ==")
maj = {}
for fix in (False, True):
    S = {arm: [{r["query_id"]: hit_h(r, fix) for r in map(json.loads, open(H / f"run{i}" / arm / f"{VH}.jsonl", encoding="utf-8"))} for i in (1, 2, 3)] for arm in ("A", "C", "D")}
    maj[fix] = {arm: {q: int(sum(s[q] for s in S[arm]) >= 2) for q in qs} for arm in S}
    print(f"  정규화 {'후' if fix else '전'}: " + " | ".join(f"{arm} 실행별 {[sum(s.values()) for s in S[arm]]} 다수결 {sum(maj[fix][arm].values())}" for arm in S))
    for x, y in (("D", "A"), ("D", "C"), ("C", "A")):
        b = sum(maj[fix][x][q] and not maj[fix][y][q] for q in qs); c = sum(maj[fix][y][q] and not maj[fix][x][q] for q in qs)
        print(f"    {x} 대 {y}(다수결) b:c {b}:{c} p={binomtest(min(b,c), b+c, 0.5).pvalue:.3f}")
for arm in ("A", "C", "D"):
    ch = [q for q in qs if maj[False][arm][q] != maj[True][arm][q]]
    print(f"  {arm} 다수결 바뀐 문항 {len(ch)}: {[(q[:8], maj[False][arm][q], maj[True][arm][q]) for q in ch]}")
print("== MultiHiertt 882 (3,072 3회) ==")
G = official_answers("train"); FD = ROOT / "results/format_dev_20261017"; VM = "R1X1F1B2D1C1E0"
from score_inclusion import gold_kind
def hit_m(r, fix):
    g = G[r["query_id"]]; fl = sf.final_line(r["raw"]); fl = norm_fl(fl) if fix else fl
    if gold_kind(g) == "yesno": return int(fl is not None and sf.yn(fl) == str(g).strip().lower())
    return int(fl is not None and included_v2(fl, g, r["kind"] == "arith"))
for d in ("final882_cap3072_qwen3_5_4b", "final882_cap3072_run2", "final882_cap3072_run3"):
    row = []
    for arm in ("A", "C", "D"):
        rows = [json.loads(l) for l in open(FD / d / arm / f"{VM}.jsonl", encoding="utf-8")]
        row.append(f"{arm} {sum(hit_m(r, False) for r in rows)} → {sum(hit_m(r, True) for r in rows)}")
    print(f"  {d}: " + " | ".join(row))
