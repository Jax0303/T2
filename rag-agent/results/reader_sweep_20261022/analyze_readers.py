"""리더 교체 실험 분석 — PREREG-2026-10-22-reader-sweep.md.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/reader_sweep_20261022/analyze_readers.py --arm D \
      --base Qwen3.5-4B=<run1>,<run2>,<run3> --new GPT-4o=<run dir> [--new 이름=dir ...]
기준(base)이 여러 실행이면 다수결(3회면 2회 이상). 채점은 score_formats2.score(마지막 줄 정답 포함 v2 = fin_inc), 정답은 공식 train.
같은 문항 정확 McNemar(binomtest), 층별 표, 사전 고정 A·D 문항(O1_wrong_classification.csv 의 A·D)과 나머지의 정답률.
"""
import argparse, csv, json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/format_dev_20261017")]
import score_formats2 as sf                                  # noqa: E402
from mh_arms import official_answers                         # noqa: E402
V = "R1X1F1B2D1C1E0"
LAYERS = ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")

def load(d, arm):
    p = Path(d) / arm / f"{V}.jsonl"
    return {r["query_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}

def mcn(x, y, qs):
    b = sum(x[q] and not y[q] for q in qs); c = sum(y[q] and not x[q] for q in qs)
    return b, c, (float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--base", required=True, help="이름=dir[,dir,...]")
    ap.add_argument("--new", nargs="+", required=True, help="이름=dir[,dir,...]")
    ap.add_argument("--ad-csv", default=str(ROOT / "results/ceiling_20261019/O1_wrong_classification.csv"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    G = official_answers("train")
    readers = [a.base] + a.new
    S, meta = {}, {}
    for spec in readers:
        name, dirs = spec.split("=", 1)
        runs = [load(d, a.arm) for d in dirs.split(",")]
        qs = set.intersection(*[set(r) for r in runs])
        per = [{q: sf.score(r[q], G[q]) for q in qs} for r in runs]
        k = len(per) // 2 + 1
        S[name] = {q: int(sum(s[q]["fin_inc"] for s in per) >= k) for q in qs}
        meta[name] = {"runs": len(per), "layer": {q: runs[0][q]["layer"] for q in qs},
                      "trunc": sum(s[q]["trunc"] for s in per for q in qs) / len(per),
                      "out_tok": sum((runs[i][q]["n_tok_out"] or 0) for i in range(len(runs)) for q in qs) / (len(per) * max(1, len(qs))),
                      "per_run": [sum(s[q]["fin_inc"] for q in qs) for s in per]}
    base = a.base.split("=")[0]
    qs = sorted(set.intersection(*[set(S[n]) for n in S]))
    N = len(qs)
    ad = {row["query_id"] for row in csv.DictReader(open(a.ad_csv, encoding="utf-8")) if row["category"].strip()[:1] in ("A", "D")}
    ad_q = [q for q in qs if q in ad]; rest_q = [q for q in qs if q not in ad]
    print(f"팔 {a.arm}, 공통 문항 {N}, A·D 고정 문항 {len(ad_q)}(파일 {len(ad)}), 나머지 {len(rest_q)}")
    print("| 리더 | 실행 | 맞힘 /N | % | " + " | ".join(LAYERS) + " | 잘림 | 출력 토큰 | 대 기준 b:c | p | A·D 문항 정답률 | 나머지 정답률 |")
    print("|---|---|---|---|" + "---|" * len(LAYERS) + "---|---|---|---|---|---|")
    out = {"arm": a.arm, "N": N, "ad_n": len(ad_q), "readers": {}}
    for n in S:
        s = S[n]; lay = meta[n]["layer"]
        tot = sum(s[q] for q in qs)
        by = {L: (sum(s[q] for q in qs if lay[q] == L), sum(1 for q in qs if lay[q] == L)) for L in LAYERS}
        b, c, p = mcn(s, S[base], qs) if n != base else (0, 0, 1.0)
        adr = sum(s[q] for q in ad_q) / max(1, len(ad_q)); rr = sum(s[q] for q in rest_q) / max(1, len(rest_q))
        print(f"| {n} | {meta[n]['runs']}{' 다수결' if meta[n]['runs'] > 1 else ''} {meta[n]['per_run']} | {tot} | {100 * tot / N:.1f} | "
              + " | ".join(f"{x}/{m}" for x, m in by.values())
              + f" | {meta[n]['trunc']:.0f} | {meta[n]['out_tok']:.0f} | {b}:{c} | {p:.3f} | {sum(s[q] for q in ad_q)}/{len(ad_q)} ({100 * adr:.0f}%) | {sum(s[q] for q in rest_q)}/{len(rest_q)} ({100 * rr:.1f}%) |")
        out["readers"][n] = {"correct": tot, "by_layer": by, "per_run": meta[n]["per_run"], "b": b, "c": c, "p": p, "ad_correct": sum(s[q] for q in ad_q), "rest_correct": sum(s[q] for q in rest_q), "items": s}
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"); print(f"[saved] {a.out}")

if __name__ == "__main__":
    main()
