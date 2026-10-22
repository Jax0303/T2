"""새 설정 비교군 표 채점 — PREREG-2026-10-23-baselines-newregime.md.
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/baselines_newregime_20261023/score_newregime.py \
      --ours "본 방법 D=<run1>,<run2>,<run3>" --arms "이름=<answer.jsonl 또는 run dir>[,<dir2>...]" ... [--out summary.json]
입력 형식 두 가지를 자동 인식: (a) prompt_sweep 실행 디렉터리(<dir>/<arm>/R1X1F1B2D1C1E0.jsonl, raw → 마지막 줄 값) (b) 비교군 답변 jsonl(`pred` 필드).
채점: 주 = 형식 허용 v2(score_v2.included_v2) + 예/아니오 규칙(score_formats2.yn), 보조 = 공식 EM(mh_exact_match). 여러 실행이면 다수결(3회면 2회 이상).
통계: 본 방법(첫 --ours) 대 각 팔 같은 문항 정확 McNemar, Holm 보정(팔 수).
"""
import argparse, json, sys
from pathlib import Path
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "results/format_dev_20261017"), str(ROOT / "results/score_fix_20261014"), str(ROOT / "results/cellform_20261012")]
import score_formats2 as sf                                  # noqa: E402
from score_v2 import included_v2                             # noqa: E402
from score_inclusion import gold_kind                        # noqa: E402
from mh_arms import official_answers                         # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match     # noqa: E402
V = "R1X1F1B2D1C1E0"
LAYERS = ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+")
G = official_answers("train")

def load_items(path):
    """{query_id: (final_answer_text, kind, layer, finished)}"""
    p = Path(path)
    if p.is_dir():
        if (p / f"{V}.jsonl").exists():                      # 팔 디렉터리를 바로 준 경우(<run>/D)
            arm_dirs = [p]
        else:
            arm_dirs = [d for d in p.iterdir() if d.is_dir() and (d / f"{V}.jsonl").exists()]
        assert len(arm_dirs) == 1, f"{p}: 팔 디렉터리 {len(arm_dirs)}개 — <run>/<arm> 으로 지정"
        rows = [json.loads(l) for l in open(arm_dirs[0] / f"{V}.jsonl", encoding="utf-8")]
        return {r["query_id"]: (sf.final_line(r["raw"]), r["kind"], r["layer"], r["finished"]) for r in rows}
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return {r["query_id"]: (r.get("pred"), r["kind"], r["layer"], True) for r in rows}

def judge(ans, q, kind):
    g = G[q]; arith = kind == "arith"
    if ans is None:
        return 0, 0
    if gold_kind(g) == "yesno":
        fin = int(sf.yn(ans) == str(g).strip().lower())
    else:
        fin = int(included_v2(ans, g, arith))
    return fin, int(mh_exact_match(ans, g, arith))

def score_spec(spec):
    name, paths = spec.split("=", 1)
    runs = [load_items(x) for x in paths.split(",")]
    qs = set.intersection(*[set(r) for r in runs])
    per = []
    for r in runs:
        per.append({q: judge(r[q][0], q, r[q][1]) for q in qs})
    k = len(per) // 2 + 1
    fin = {q: int(sum(s[q][0] for s in per) >= k) for q in qs}
    em = {q: int(sum(s[q][1] for s in per) >= k) for q in qs}
    layer = {q: runs[0][q][2] for q in qs}
    return name, fin, em, layer, [sum(s[q][0] for q in qs) for s in per]

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0] * m; cur = 0
    for rank, i in enumerate(order):
        cur = max(cur, min(1.0, (m - rank) * ps[i])); adj[i] = cur
    return adj

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ours", required=True); ap.add_argument("--arms", nargs="*", default=[]); ap.add_argument("--out", default="")
    a = ap.parse_args()
    specs = [score_spec(a.ours)] + [score_spec(s) for s in a.arms]
    qs = sorted(set.intersection(*[set(s[1]) for s in specs])); N = len(qs)
    base = specs[0]
    print(f"공통 문항 {N}")
    print("| 팔 | 실행별 맞힘 | 맞힘 /N(다수결) | % | " + " | ".join(LAYERS) + " | 공식 EM | 대 본 방법 b:c | p | Holm p |")
    print("|---|---|---|---|" + "---|" * len(LAYERS) + "---|---|---|---|")
    stats = []
    for name, fin, em, layer, per_run in specs:
        b = sum(base[1][q] and not fin[q] for q in qs); c = sum(fin[q] and not base[1][q] for q in qs)
        p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        stats.append((name, fin, em, layer, per_run, b, c, p))
    ps = [s[7] for s in stats[1:]]; adj = holm(ps) if ps else []
    out = {"N": N, "arms": {}}
    for i, (name, fin, em, layer, per_run, b, c, p) in enumerate(stats):
        tot = sum(fin[q] for q in qs); emt = sum(em[q] for q in qs)
        by = [f"{sum(fin[q] for q in qs if layer[q] == L)}/{sum(1 for q in qs if layer[q] == L)}" for L in LAYERS]
        hp = "—" if i == 0 else f"{adj[i - 1]:.3f}"
        print(f"| {name} | {per_run} | {tot} | {100 * tot / N:.1f} | " + " | ".join(by) + f" | {emt} ({100 * emt / N:.1f}) | {b}:{c} | {p:.3f} | {hp} |")
        out["arms"][name] = {"correct": tot, "em": emt, "per_run": per_run, "b": b, "c": c, "p": p, "holm_p": None if i == 0 else adj[i - 1], "items": {q: fin[q] for q in qs}}
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8"); print(f"[saved] {a.out}")

if __name__ == "__main__":
    main()
