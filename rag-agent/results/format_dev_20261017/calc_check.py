"""오프라인 분석: 모델이 쓴 '식 = 값' 중 최종 답과 이어진 식을 다시 계산해 보면(계산기 검증 단계) 몇 문항이 바뀌는가. LLM 호출 없음.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/calc_check.py <run-dir> [...] --gold train
"""
import json, math, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
import score_formats2 as sf
from score_v2 import included_v2
from score_inclusion import gold_kind
from mh_arms import official_answers
G = official_answers("train") if "--gold" in sys.argv else {r["query_id"]: r["answer"] for r in map(json.loads, open(ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl"))}
dirs = [a for a in sys.argv[1:] if not a.startswith("--") and a != "train"]
NUM = r"-?\$?\(?-?[\d,]*\.?\d+\)?%?"
PAIR = re.compile(r"([\(\)\d,\.\s\+\-\*/×÷x\$%]{3,}?)\s*(?:=|≈|\\approx)\s*(" + NUM + r")")
def tof(s):
    s = s.strip().replace("$", "").replace(",", ""); pct = s.endswith("%"); s = s.rstrip("%")
    neg = s.startswith("(") and s.endswith(")"); s = s.strip("()")
    try: v = float(s)
    except ValueError: return None, False
    return (-v if neg else v), pct
def ev(expr):
    e = expr.replace("$", "").replace(",", "").replace("×", "*").replace("÷", "/").replace("x", "*").replace("%", "")
    e = re.sub(r"\(\s*(-?\d*\.?\d+)\s*\)", r"\1", e)
    if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)]+", e) or not re.search(r"\d\s*[\+\-\*/]\s*[\(\d]", e): return None
    try: return float(eval(e, {"__builtins__": {}}, {}))
    except Exception: return None
def close(a, b, tol=1e-3): return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))
for d in dirs:
    print(f"\n== {Path(d).name} ==")
    for arm in ("D", "A", "C"):
        p = Path(d) / arm / "R1X1F1B2D1C1E0.jsonl"
        if not p.exists(): continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        n_fin = n_pair = n_slip = w2r = r2w = 0; ex = []
        for r in rows:
            g = G[r["query_id"]]; arith = r["kind"] == "arith"
            if gold_kind(g) != "num": continue
            fl = sf.final_line(r["raw"])
            if fl is None: continue
            fv, fpct = tof(fl.split()[0] if fl.split() else "")
            if fv is None: continue
            n_fin += 1
            body = r["raw"][: r["raw"].lower().rfind("final answer")]
            cand = None
            for m in PAIR.finditer(body):
                shown, spct = tof(m.group(2)); val = ev(m.group(1))
                if shown is None or val is None: continue
                scale = 1.0
                if close(shown, fv): scale = 1.0
                elif close(shown * 100, fv): scale = 100.0
                elif close(shown / 100, fv): scale = 0.01
                else: continue
                cand = (m.group(1).strip(), shown, val, scale)
            if cand is None: continue
            n_pair += 1
            expr, shown, val, scale = cand
            if close(val, shown, 2e-3): continue
            n_slip += 1
            new_final = f"{val * scale:.6g}"
            before = int(included_v2(fl, g, arith)); after = int(included_v2(new_final, g, arith))
            w2r += (not before and after); r2w += (before and not after)
            if len(ex) < 3 and (before != after): ex.append((r["query_id"][:8], expr[:50], shown, round(val, 4), g, before, after))
        print(f"  {arm}: 숫자 정답·답 줄 있음 {n_fin}, 최종 답과 이어진 식 찾음 {n_pair}, 식 재계산이 모델 값과 다름(계산 실수) {n_slip} → 틀림→맞음 {w2r}, 맞음→틀림 {r2w}, 순이득 {w2r - r2w}")
        for e in ex: print("     예:", e)
