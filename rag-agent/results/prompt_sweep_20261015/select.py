"""3단계 선택 — PREREG-2026-10-15-prompt-sweep.md. score.json 에서 P*(S 최대, 동률이면 출력 토큰 평균 적은 쪽), P_C(C 의 W 최대), P_A(A 의 W 최대).
  .venv/bin/python results/prompt_sweep_20261015/select.py results/prompt_sweep_20261015/sweep_qwen3_5_4b/score.json
"""
import json, sys
d = json.load(open(sys.argv[1]))
rows = [r for r in d["rows"] if r["S"] is not None and all(not r["arms"][a].get("incomplete") for a in ("A", "C"))]
tok = lambda r: r["arms"]["A"]["out_tok"] + r["arms"]["C"]["out_tok"]
best = sorted(rows, key=lambda r: (-r["S"], tok(r)))
pc = sorted(rows, key=lambda r: (-r["arms"]["C"]["W"], r["arms"]["C"]["out_tok"]))
pa = sorted(rows, key=lambda r: (-r["arms"]["A"]["W"], r["arms"]["A"]["out_tok"]))
print(f"완료 변형 {len(rows)}")
print(f"P*  = {best[0]['variant']}  S={best[0]['S']:.4f}  (2위 {best[1]['variant']} {best[1]['S']:.4f}, 3위 {best[2]['variant']} {best[2]['S']:.4f})")
print(f"P_C = {pc[0]['variant']}  W_C={pc[0]['arms']['C']['W']:.4f} 맞힘 {pc[0]['arms']['C']['inc']}  (P* 의 C: {best[0]['arms']['C']['W']:.4f}/{best[0]['arms']['C']['inc']})")
print(f"P_A = {pa[0]['variant']}  W_A={pa[0]['arms']['A']['W']:.4f} 맞힘 {pa[0]['arms']['A']['inc']}  (P* 의 A: {best[0]['arms']['A']['W']:.4f}/{best[0]['arms']['A']['inc']})")
json.dump({"P*": best[0]["variant"], "P_C": pc[0]["variant"], "P_A": pa[0]["variant"], "n_variants": len(rows),
           "top3": [(r["variant"], r["S"]) for r in best[:3]]}, open(sys.argv[1].replace("score.json", "selection.json"), "w"), indent=1)
