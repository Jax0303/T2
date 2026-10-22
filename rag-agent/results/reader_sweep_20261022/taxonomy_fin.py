"""오류 원인 자동 분류(마지막 줄 채점 레코드용) — PREREG-2026-10-22-reader-sweep.md §분석 3. taxonomy_auto.py(옛 cot 레코드용)를 fin_inc 형식에 맞게 다시 씀.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/reader_sweep_20261022/taxonomy_fin.py --arm D --readers Qwen3.5-4B=<run dir> GPT-4o=<run dir> [--examples 6]
리더마다 실행 1개(여러 개면 첫 것). 틀림 = score_formats2.score 의 fin_inc == 0. 분류는 규칙만 쓰고 사람 판정은 없다.
A·D 고정 문항(O1_wrong_classification.csv 의 A·D 100)은 따로 센다 — 그 문항의 틀림은 10-19 사람 판정으로 "데이터 쪽"이다.
"""
import argparse, csv, json, math, random, re, sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/format_dev_20261017")]
import score_formats2 as sf                                  # noqa: E402
from mh_arms import official_answers                         # noqa: E402
from score_inclusion import gold_kind                        # noqa: E402
V = "R1X1F1B2D1C1E0"
LAYERS = ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")
NUM = re.compile(r"\(?-?\$?\s*\d[\d,]*\.?\d*\)?%?")

def num(s):
    if isinstance(s, (int, float)): return float(s)
    m = NUM.search(str(s))
    if not m: return None
    x = m.group(0); neg = x.startswith("(") and x.endswith(")") or x.startswith("-") or "-" in x[:3]
    x = re.sub(r"[^\d.]", "", x)
    try: v = float(x)
    except ValueError: return None
    return -v if neg else v

def classify(r, g, s):
    arith = r["kind"] == "arith"; fl = sf.final_line(r["raw"])
    if not r["finished"]: return "T 잘림"
    if fl is None: return "A 마지막 줄 없음"
    if not fl.strip(): return "B 빈 답"
    if re.search(r"cannot|can't|not (provided|available|possible|enough|determin|specified)|insufficient|unknown|n/a|no information", fl, re.I): return "C 답 거부"
    if gold_kind(g) == "yesno": return "Y 예/아니오 반대"
    if s["inc"]: return "F 풀이 본문에 정답 값 등장(마지막 줄은 다름; 중간값일 수 있음)"
    gl = g if isinstance(g, list) else [g]
    if len(gl) > 1: return "I 다중 정답"
    gn, pn = num(g), num(fl)
    if arith:
        if pn is None: return "E 숫자 아닌 답(산술)"
        if gn is None: return "E2 정답이 숫자 아님(산술)"
        if gn != 0 and math.isclose(abs(gn), abs(pn), rel_tol=1e-6): return "D 부호만 다름"
        if gn != 0 and abs(pn - gn) / abs(gn) < 0.02: return "G 2% 이내(반올림)"
        if gn and pn:
            ratio = abs(pn / gn)
            for k in (10, 100, 1000, 10000, 100000, 1e6):
                if math.isclose(ratio, k, rel_tol=1e-3) or math.isclose(ratio, 1 / k, rel_tol=1e-3): return "H 10^n 척도 차이"
        return "Z 계산·셀 선택(잔여)"
    if pn is not None and gn is not None:
        if math.isclose(pn, gn, rel_tol=1e-6): return "J 숫자 같음, 문자열 불일치"
        if gn != 0 and abs(pn - gn) / abs(gn) < 0.02: return "G 2% 이내(반올림)"
        return "Z 다른 셀 값(잔여)"
    if pn is None and gn is not None: return "K 정답 숫자, 답은 텍스트"
    if pn is not None and gn is None: return "L 정답 텍스트, 답은 숫자"
    return "Z 텍스트 불일치(잔여)"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--readers", nargs="+", required=True)
    ap.add_argument("--ad-csv", default=str(ROOT / "results/ceiling_20261019/O1_wrong_classification.csv")); ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args(); random.seed(0)
    G = official_answers("train")
    ad = {row["query_id"] for row in csv.DictReader(open(a.ad_csv, encoding="utf-8")) if row["category"].strip()[:1] in ("A", "D")}
    for spec in a.readers:
        name, d = spec.split("=", 1); d = d.split(",")[0]
        rows = [json.loads(l) for l in open(Path(d) / a.arm / f"{V}.jsonl", encoding="utf-8")]
        wrong = []
        for r in rows:
            s = sf.score(r, G[r["query_id"]])
            if not s["fin_inc"]: wrong.append((r, G[r["query_id"]], s))
        by = collections.defaultdict(collections.Counter); inad = collections.Counter()
        for r, g, s in wrong:
            c = classify(r, g, s); by[r["layer"]][c] += 1; inad[r["query_id"] in ad] += 1
        print(f"\n### {name} — 팔 {a.arm}, 문항 {len(rows)}, 틀림 {len(wrong)} (A·D 고정 100문항 안 {inad[True]}, 밖 {inad[False]})")
        cats = sorted({c for v in by.values() for c in v})
        print("| 분류 | " + " | ".join(LAYERS) + " | 합 | % |"); print("|---|" + "---:|" * (len(LAYERS) + 2))
        for c in cats:
            vals = [by[L][c] for L in LAYERS]; print(f"| {c} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} | {100 * sum(vals) / len(wrong):.0f} |")
        print("| 합 | " + " | ".join(str(sum(by[L].values())) for L in LAYERS) + f" | {len(wrong)} | 100 |")
        if a.examples:
            res = [(r, g, s) for r, g, s in wrong if classify(r, g, s).startswith("Z") and r["query_id"] not in ad]
            print(f"잔여(Z, A·D 밖) 예시 {min(a.examples, len(res))}/{len(res)}:")
            for r, g, s in random.sample(res, min(a.examples, len(res))):
                print(f"- [{r['layer']}] {r['query_id'][:8]} | gold={g!r} | 마지막 줄={sf.final_line(r['raw'])!r} | 풀이 끝: {r['raw'][-160:].replace(chr(10), ' / ')!r}")

if __name__ == "__main__":
    main()
