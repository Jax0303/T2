#!/usr/bin/env python3
"""Phase 2 요약: phase2_retrieval.json → phase2_baselines.csv + phase2_summary.md.
형용사 금지, 수치만."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "results" / "phase2_retrieval.json").read_text())

rows = []
for name, m in res["overall"].items():
    rows.append({"method": name, "r1": m["r1"], "r5": m["r5"], "r10": m["r10"],
                 "mrr": m["mrr"], "ndcg": m["ndcg"]})

csv_path = ROOT / "results" / "phase2_baselines.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "r1", "r5", "r10", "mrr", "ndcg"])
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

cfg = res["config"]
lines = ["# Phase 2 — 검색 베이스라인 요약 (라우팅 없음)", ""]
lines.append(f"풀={cfg['pool_size']}표 · 평가={cfg['n_eval']} {cfg['split']} 질의 · "
             f"임베더={cfg['embedder']} · seed={cfg['seed']}")
lines.append(f"best dense 직렬화: `{cfg.get('best_dense_serializer')}` · "
             f"프로토콜: {cfg['protocol']}")
lines.append("")
lines.append("| method | R@1 | R@5 | R@10 | MRR | nDCG@10 |")
lines.append("|---|---|---|---|---|---|")
order = sorted(rows, key=lambda r: -r["mrr"])
for r in order:
    lines.append(f"| {r['method']} | {r['r1']:.4f} | {r['r5']:.4f} | {r['r10']:.4f} "
                 f"| {r['mrr']:.4f} | {r['ndcg']:.4f} |")
lines.append("")
# BM25 grid 증거
bm = res["overall"]["bm25"]
lines.append(f"BM25 튜닝(grid k1×b): 채택 k1={bm['best_k1']}, b={bm['best_b']} "
             f"(전체 grid는 results/phase2_retrieval.json::grids.bm25).")
lines.append("")
# paired stats
lines.append("## R@1 paired bootstrap vs best-dense (1000 resample, 95% CI)")
lines.append("| method | Δ(method−best_dense) | CI95 | sig |")
lines.append("|---|---|---|---|")
for name, s in res["paired_r1_vs_best_dense"].items():
    lines.append(f"| {name} | {s['delta']:+.4f} | [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] | {s['sig']} |")
lines.append("")
# 1줄 해석 (수치만)
best = order[0]
lines.append(f"해석(수치): 최고 MRR = `{best['method']}` ({best['mrr']:.4f}); "
             f"BM25(튜닝) MRR={res['overall']['bm25']['mrr']:.4f}, "
             f"dense_plain_markdown MRR={res['overall'].get('dense_plain_markdown',{}).get('mrr',float('nan')):.4f}, "
             f"hybrid_rrf MRR={res['overall']['hybrid_rrf']['mrr']:.4f}.")

# ---- answer-side (if available) ----
ans_path = ROOT / "results" / "phase2_answers.json"
if ans_path.exists():
    a = json.loads(ans_path.read_text())
    lines.append("")
    lines.append("## 답변 측 end-to-end (top-1 표 → LLM)")
    lines.append(f"LLM={a['config']['llm']} · n={a['config']['n_eval']} · context={a['config']['context']}")
    lines.append("")
    lines.append("| baseline | R@1 | EM | NM | F1 |")
    lines.append("|---|---|---|---|---|")
    border = {"oracle": "상한", "nocontext": "하한"}
    for b in ["oracle", "dense_header_path", "bm25", "nocontext"]:
        if b in a["overall"]:
            m = a["overall"][b]
            tag = f" ({border[b]})" if b in border else ""
            lines.append(f"| {b}{tag} | {m['R@1']:.3f} | {m['EM']:.3f} | {m['NM']:.3f} | {m['F1']:.3f} |")
    # gate checks
    nm = {b: a["overall"][b]["NM"] for b in a["overall"]}
    retr = [b for b in nm if b not in ("oracle", "nocontext")]
    g_up = all(nm["oracle"] >= nm[b] for b in retr)
    g_lo = all(nm["nocontext"] <= nm[b] for b in retr)
    lines.append("")
    lines.append(f"Gate 2 상한(oracle≥검색): {g_up} · 하한(nocontext≤검색): {g_lo}")
    # gap
    g = a["gap"]
    lines.append("")
    lines.append("## retrieval–answer gap")
    lines.append(f"across-baseline Spearman ρ(R@1, NM) = {g['across_baseline_spearman_rho']:.3f} (p={g['p']:.3f})")
    for b, c in g["per_query_conditional"].items():
        ph = c["P_correct_given_hit"]; pm = c["P_correct_given_miss"]
        lines.append(f"- {b}: P(정답|top1 적중)={ph:.3f} (n={c['n_hit']}) vs "
                     f"P(정답|불일치)={pm:.3f} (n={c['n_miss']})")

(ROOT / "results" / "phase2_summary.md").write_text("\n".join(lines))
print("wrote results/phase2_baselines.csv and results/phase2_summary.md")
print("\n".join(lines))
