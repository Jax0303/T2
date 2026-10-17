"""전달 형식 dev 채점 — PREREG-2026-10-17-delivery-format.md (결과 전에 작성).
주 채점 = 마지막 'Final answer:' 줄만 정답 포함 v2(예/아니오는 그 줄의 첫 yes/no; 줄 없으면 0).
보조 = 출력 전체 정답 포함 v2(예/아니오 첫 yes/no 낱말, 판정표 없음 — 지시문 탐색 dev 규칙), 공식 EM, 잘림, 출력 토큰.
쌍 비교 = 같은 문항 정확 McNemar(이항검정), 각 팔 대 C·대 A. 결과는 <out-dir>/score.json·표준출력.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_formats.py <out-dir> [--variant R1X1F1B2D1C1E0]
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
from answer_accuracy_mh import extract                      # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match    # noqa: E402
from score_inclusion import gold_kind                       # noqa: E402
from score_v2 import included_v2                            # noqa: E402
GOLD = {r["query_id"]: r["answer"] for r in map(json.loads, open(ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl", encoding="utf-8"))}
W_LOOKUP, W_ARITH = 511 / 882, 371 / 882
yn = lambda s: (re.search(r"\b(yes|no)\b", (s or "").lower()) or [None, None])[1] if re.search(r"\b(yes|no)\b", (s or "").lower()) else None
def final_line(raw):
    m = list(re.finditer(r"final answer\s*[:：]\s*(.*)", raw or "", re.I)); return m[-1].group(1).strip() if m else None
def score(r):
    g, arith = GOLD[r["query_id"]], r["kind"] == "arith"; raw = r["raw"]; fl = final_line(raw)
    if gold_kind(g) == "yesno":
        inc = int(yn(raw) == str(g).strip().lower()); fin_inc = int(fl is not None and yn(fl) == str(g).strip().lower())
    else:
        inc = int(included_v2(raw, g, arith)); fin_inc = int(fl is not None and included_v2(fl, g, arith))
    pred, _ = extract(raw)
    return {"inc": inc, "fin_inc": fin_inc, "em": int(mh_exact_match(pred, g, arith)), "trunc": int(not r["finished"]), "n_out": r["n_tok_out"],
            "kind": r["kind"], "layer": r["layer"], "no_final": int(fl is None)}
def mcnemar(a, b, qs, k):
    bb = sum(a[q][k] and not b[q][k] for q in qs); cc = sum(b[q][k] and not a[q][k] for q in qs)
    return {"b": bb, "c": cc, "p": float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--variant", default="R1X1F1B2D1C1E0"); a = ap.parse_args()
    D = Path(a.out_dir); S = {}
    for arm_dir in sorted(p for p in D.iterdir() if p.is_dir()):
        p = arm_dir / f"{a.variant}.jsonl"
        if p.exists():
            S[arm_dir.name] = {r["query_id"]: score(r) for r in map(json.loads, open(p, encoding="utf-8"))}
    qs = sorted(GOLD); out = {"n": len(qs), "arms": {}, "pairs": {}}
    print(f"| 팔 | 문항 | 마지막 줄 맞힘 | % | 조회 | 산술 | W(마지막 줄) | 전체 포함 | 공식 EM | 잘림 | 답 줄 없음 | 출력 토큰 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, s in S.items():
        if len(s) < len(qs): print(f"| {arm} | {len(s)}/{len(qs)} 진행 중 |"); continue
        v = [s[q] for q in qs]; lk = [x for x in v if x["kind"] == "lookup"]; ar = [x for x in v if x["kind"] == "arith"]
        W = W_LOOKUP * sum(x["fin_inc"] for x in lk) / len(lk) + W_ARITH * sum(x["fin_inc"] for x in ar) / len(ar)
        row = {"fin_inc": sum(x["fin_inc"] for x in v), "pct": round(100 * sum(x["fin_inc"] for x in v) / len(v), 1), "lookup": f"{sum(x['fin_inc'] for x in lk)}/{len(lk)}",
               "arith": f"{sum(x['fin_inc'] for x in ar)}/{len(ar)}", "W": round(W, 4), "inc": sum(x["inc"] for x in v), "em": round(sum(x["em"] for x in v) / len(v), 4),
               "trunc": sum(x["trunc"] for x in v), "no_final": sum(x["no_final"] for x in v), "out_tok": round(sum(x["n_out"] for x in v) / len(v), 1),
               "by_layer_fin": {L: sum(x["fin_inc"] for x in v if x["layer"] == L) for L in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")}}
        out["arms"][arm] = row
        print(f"| {arm} | {len(v)} | {row['fin_inc']} | {row['pct']} | {row['lookup']} | {row['arith']} | {row['W']} | {row['inc']} | {row['em']} | {row['trunc']} | {row['no_final']} | {row['out_tok']} |")
    full = [arm for arm in S if len(S[arm]) == len(qs)]
    for arm in full:
        for ref in ("C", "A"):
            if ref in full and arm != ref:
                for k in ("fin_inc", "inc"):
                    r = mcnemar(S[arm], S[ref], qs, k); out["pairs"][f"{arm}_vs_{ref}_{k}"] = r
                    print(f"{arm} 대 {ref} [{k}]: {sum(S[arm][q][k] for q in qs)} 대 {sum(S[ref][q][k] for q in qs)}, b:c {r['b']}:{r['c']} p={r['p']:.3f}")
    json.dump(out, open(D / "score.json", "w"), ensure_ascii=False, indent=1)
if __name__ == "__main__":
    main()
