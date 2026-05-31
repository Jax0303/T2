#!/home/user/T2/hart-table-retrieval/.venv/bin/python3
"""Run multiple ablation conditions in one process (model loaded once)."""
import os, sys, json, time
os.environ.setdefault("LLM_BACKEND", "local")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codegen_eval import run_pipeline

CONDITIONS = [
    # (label, per_class, ablation)
    ("gold_ceiling", 10, "gold-table-codegen"),
    ("adaptive",     20, "adaptive"),
    ("always_code",  20, "always-codegen"),
]

summary = []
for label, per_class, ablation in CONDITIONS:
    print("\n" + "#" * 72)
    print(f"# {label}  ablation={ablation}  per_class={per_class}")
    print("#" * 72)
    t0 = time.time()
    out = run_pipeline(per_class=per_class, ablation=ablation, verbose=False,
                      out_name=f"ablation_{label}.json")
    elapsed = time.time() - t0
    summary.append({
        "label": label, "ablation": ablation, "per_class": per_class,
        "n": out["n"], "nm": out["nm_rate"], "ci": out["ci95"],
        "class_stats": out["class_stats"],
        "elapsed_min": round(elapsed/60, 2),
    })

print("\n" + "=" * 72)
print("FINAL COMPARISON")
print("=" * 72)
print(f"{'Label':16s} {'Ablation':22s} {'N':>4s} {'NM':>6s}  {'95% CI':>16s}  {'time':>6s}")
print("-" * 78)
for s in summary:
    ci = f"[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]"
    print(f"{s['label']:16s} {s['ablation']:22s} {s['n']:4d} {s['nm']:6.3f}  {ci:>16s}  {s['elapsed_min']:5.1f}m")

print("\nPer-class NM:")
classes = ["multi_op_formula","arithmetic_agg","pair_or_topk_arg","single_arg","comparison_or_count"]
print(f"{'class':26s} " + " ".join(f"{s['label']:>14s}" for s in summary))
for cls in classes:
    row = [f"{cls:26s}"]
    for s in summary:
        st = s["class_stats"].get(cls, {})
        n = st.get("n", 0); c = st.get("correct", 0)
        row.append(f"{c}/{n}".rjust(14) if n else "  -          ")
    print(" ".join(row))

out_path = "/home/user/T2-1/rag-agent/results/ablations_summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSaved: {out_path}")
