"""882 확인 채점 — PREREG-2026-10-17-delivery-format-882.md. 규칙은 score_formats.py(고정)와 같고 정답만 공식 train 으로 바꾼다.
예/아니오 판정표는 쓰지 않는다(마지막 줄 yes/no 는 명시적; 전체 포함 보조 채점은 첫 yes/no 낱말 규칙) — 오전 882 결과(판정표 16건 반영)와는 그 점이 다르므로
오전 출력도 같은 규칙으로 다시 채점해 나란히 적는다.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_882.py <out-dir> [--ref-dir results/prompt_sweep_20261015/final_882]
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts")]
import score_formats as sf                                  # noqa: E402
from mh_arms import official_answers                        # noqa: E402
sf.GOLD = official_answers("train")
V = "R1X1F1B2D1C1E0"
def load_dir(D):
    S = {}
    for arm_dir in sorted(p for p in Path(D).iterdir() if p.is_dir()):
        p = arm_dir / f"{V}.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if len(rows) == 882: S[arm_dir.name] = {r["query_id"]: sf.score(r) for r in rows}
            else: print(f"  {arm_dir.name}: {len(rows)}/882 진행 중")
    return S
def table(S, label):
    qs = sorted(next(iter(S.values()))) if S else []
    print(f"\n== {label} ==\n| 팔 | 마지막 줄 맞힘 /882 | % | 조회 /511 | 산술 /371 | 전체 포함 | 공식 EM | 잘림 | 답 줄 없음 | 출력 토큰 |\n|---|---|---|---|---|---|---|---|---|---|")
    out = {}
    for arm, s in S.items():
        v = [s[q] for q in qs]; lk = [x for x in v if x["kind"] == "lookup"]; ar = [x for x in v if x["kind"] == "arith"]
        out[arm] = {"fin_inc": sum(x["fin_inc"] for x in v), "lookup": sum(x["fin_inc"] for x in lk), "arith": sum(x["fin_inc"] for x in ar), "inc": sum(x["inc"] for x in v),
                    "em": round(sum(x["em"] for x in v) / len(v), 4), "trunc": sum(x["trunc"] for x in v), "no_final": sum(x["no_final"] for x in v), "out_tok": round(sum(x["n_out"] for x in v) / len(v), 1),
                    "by_layer_fin": {L: sum(x["fin_inc"] for x in v if x["layer"] == L) for L in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")}}
        o = out[arm]; print(f"| {arm} | {o['fin_inc']} | {100*o['fin_inc']/882:.1f} | {o['lookup']} | {o['arith']} | {o['inc']} | {o['em']} | {o['trunc']} | {o['no_final']} | {o['out_tok']} |")
    return out, qs
def pairs(S, qs, specs):
    res = {}
    for a, b in specs:
        if a in S and b in S:
            for k in ("fin_inc", "inc"):
                r = sf.mcnemar(S[a], S[b], qs, k); res[f"{a}_vs_{b}_{k}"] = r
                for kind in ("lookup", "arith"):
                    sub = [q for q in qs if S[a][q]["kind"] == kind]; r[kind] = sf.mcnemar(S[a], S[b], sub, k)
                print(f"{a} 대 {b} [{k}]: {sum(S[a][q][k] for q in qs)} 대 {sum(S[b][q][k] for q in qs)}, b:c {r['b']}:{r['c']} p={r['p']:.4f} | 조회 {r['lookup']['b']}:{r['lookup']['c']} p={r['lookup']['p']:.3f} 산술 {r['arith']['b']}:{r['arith']['c']} p={r['arith']['p']:.3f}")
    return res
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--ref-dir", default=str(ROOT / "results/prompt_sweep_20261015/final_882")); a = ap.parse_args()
    S = load_dir(a.out_dir); out = {"arms": {}, "pairs": {}}
    if S:
        out["arms"], qs = table(S, Path(a.out_dir).name); out["pairs"] = pairs(S, qs, [("D", "A"), ("D", "C"), ("C", "A")])
    R = load_dir(a.ref_dir)
    if R:
        out["ref_arms"], qs = table(R, "오전 882(같은 규칙으로 재채점, 판정표 없음)"); out["ref_pairs"] = pairs(R, qs, [("C", "A")])
        if S:
            print("\n== 같은 조건 재실행 차이(오전 대 지금) ==")
            for arm in ("C", "A"):
                if arm in S and arm in R:
                    r = sf.mcnemar(S[arm], R[arm], qs, "fin_inc"); print(f"  {arm}: 지금 {sum(S[arm][q]['fin_inc'] for q in qs)} 대 오전 {sum(R[arm][q]['fin_inc'] for q in qs)}, b:c {r['b']}:{r['c']}"); out["pairs"][f"{arm}_now_vs_morning_fin_inc"] = r
    json.dump(out, open(Path(a.out_dir) / "score.json", "w"), ensure_ascii=False, indent=1)
if __name__ == "__main__":
    main()
