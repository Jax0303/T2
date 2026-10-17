"""형식 비교 채점 v2 — PREREG-2026-10-18-table-cap.md. score_formats.py 와 같되 (1) 마지막 줄 라벨 정규식이 굵은 글씨를 허용한다
(`final answer\**\s*[:：]`; v1 은 `**Final answer**:` 를 놓쳐 팔마다 2~8건 틀림 처리했다), (2) 여러 실행(out-dir 여러 개)을 받아 실행별 표와 평균을 낸다.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_formats2.py <out-dir> [<out-dir2> ...] [--gold train] [--variant V]
"""
import argparse, json, re, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/cellform_20261012"), str(ROOT / "results/score_fix_20261014")]
from answer_accuracy_mh import extract                      # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match    # noqa: E402
from score_inclusion import gold_kind                       # noqa: E402
from score_v2 import included_v2                            # noqa: E402
yn = lambda s: (lambda m: m.group(1) if m else None)(re.search(r"\b(yes|no)\b", (s or "").lower()))
def final_line(raw):
    m = list(re.finditer(r"final answer\**\s*[:：]\s*(.*)", raw or "", re.I)); return m[-1].group(1).strip() if m else None
def score(r, g):
    arith = r["kind"] == "arith"; raw = r["raw"]; fl = final_line(raw)
    if gold_kind(g) == "yesno":
        inc = int(yn(raw) == str(g).strip().lower()); fin = int(fl is not None and yn(fl) == str(g).strip().lower())
    else:
        inc = int(included_v2(raw, g, arith)); fin = int(fl is not None and included_v2(fl, g, arith))
    pred, _ = extract(raw)
    return {"inc": inc, "fin_inc": fin, "em": int(mh_exact_match(pred, g, arith)), "trunc": int(not r["finished"]), "n_out": r["n_tok_out"], "kind": r["kind"], "layer": r["layer"], "no_final": int(fl is None)}
def mcnemar(a, b, qs, k="fin_inc"):
    bb = sum(a[q][k] and not b[q][k] for q in qs); cc = sum(b[q][k] and not a[q][k] for q in qs)
    return {"b": bb, "c": cc, "p": float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--gold", default=""); ap.add_argument("--variant", default="R1X1F1B2D1C1E0"); a = ap.parse_args()
    if a.gold:
        from mh_arms import official_answers; G = official_answers(a.gold)
    else:
        G = {r["query_id"]: r["answer"] for r in map(json.loads, open(ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl", encoding="utf-8"))}
    runs = {}
    for d in a.dirs:
        S = {}
        for arm_dir in sorted(p for p in Path(d).iterdir() if p.is_dir()):
            p = arm_dir / f"{a.variant}.jsonl"
            if p.exists():
                rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
                if rows: S[arm_dir.name] = {r["query_id"]: score(r, G[r["query_id"]]) for r in rows}
        runs[Path(d).name] = S
    qs = sorted(G); N = len(qs); out = {"n": N, "runs": {}, "mean": {}, "pairs": {}}
    arms = sorted({arm for S in runs.values() for arm in S})
    for rn, S in runs.items():
        out["runs"][rn] = {}
        print(f"\n== {rn} ==\n| 팔 | 마지막 줄 맞힘 /{N} | % | 조회 | 산술 | 전체 포함 | 공식 EM | 잘림 | 출력 토큰 |\n|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            if arm not in S: continue
            s = S[arm]
            if len(s) < N: print(f"| {arm} | {len(s)}/{N} 진행 중 |"); continue
            v = [s[q] for q in qs]; lk = [x for x in v if x["kind"] == "lookup"]; ar = [x for x in v if x["kind"] == "arith"]
            row = {"fin_inc": sum(x["fin_inc"] for x in v), "lookup": sum(x["fin_inc"] for x in lk), "arith": sum(x["fin_inc"] for x in ar), "inc": sum(x["inc"] for x in v), "em": round(sum(x["em"] for x in v) / N, 4), "trunc": sum(x["trunc"] for x in v), "out_tok": round(sum(x["n_out"] for x in v) / N, 1)}
            out["runs"][rn][arm] = row
            print(f"| {arm} | {row['fin_inc']} | {100*row['fin_inc']/N:.1f} | {row['lookup']}/{len(lk)} | {row['arith']}/{len(ar)} | {row['inc']} | {row['em']} | {row['trunc']} | {row['out_tok']} |")
    complete = {arm for arm in arms if all(arm in S and len(S[arm]) == N for S in runs.values())}
    if len(runs) > 1 and complete:
        print("\n== 실행 평균(마지막 줄 맞힘) ==")
        for arm in sorted(complete):
            vals = [out["runs"][rn][arm]["fin_inc"] for rn in runs]; out["mean"][arm] = sum(vals) / len(vals)
            print(f"  {arm}: {vals} 평균 {out['mean'][arm]:.1f}")
    print("\n== 쌍 비교(마지막 줄, 실행별) ==")
    refs = [r for r in ("D", "A", "C") if r in complete]
    for rn, S in runs.items():
        for arm in sorted(complete):
            for ref in refs:
                if arm == ref: continue
                r = mcnemar(S[arm], S[ref], qs); out["pairs"][f"{rn}:{arm}_vs_{ref}"] = r
                print(f"  {rn} {arm} 대 {ref}: {sum(S[arm][q]['fin_inc'] for q in qs)} 대 {sum(S[ref][q]['fin_inc'] for q in qs)} b:c {r['b']}:{r['c']} p={r['p']:.3f}")
    json.dump(out, open(Path(a.dirs[0]).parent / f"score2_{'_'.join(Path(d).name for d in a.dirs)}.json", "w"), ensure_ascii=False, indent=1)
if __name__ == "__main__":
    main()
