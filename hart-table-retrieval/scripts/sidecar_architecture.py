"""Render the sidecar-verifier agent architecture as a PNG diagram.

Output: docs/sidecar_architecture.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, text, color, fontsize=10, fc=None):
    fc = fc or color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor=color, facecolor=fc, alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="black", wrap=True)


def _arrow(ax, x1, y1, x2, y2, label=None, color="#444", lw=1.4, ls="-"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, linestyle=ls,
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.08, label, ha="center", fontsize=8, color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title(
        "Sidecar Verifier Agent — Vector RAG + Structured Table Cross-Verification",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Color palette
    C_QUERY = "#1f77b4"
    C_VEC   = "#d62728"
    C_TBL   = "#2ca02c"
    C_AGENT = "#9467bd"
    C_OUT   = "#ff7f0e"

    # ---- INPUT ----
    _box(ax, 0.3, 4.0, 1.8, 0.9, "User Query", C_QUERY, 11, fc="#dde9f5")

    # ---- OFFLINE / INDEX ----
    ax.text(3.4, 7.7, "OFFLINE (one-time indexing)", fontsize=10,
            color="#666", style="italic")
    _box(ax, 2.6, 6.6, 3.2, 0.7, "HiTab / FeTaQA / TabFact / OTTQA …", "#666", 10, fc="#f0f0f0")
    _box(ax, 2.6, 5.4, 1.4, 0.9, "Serializer\n(markdown)", C_VEC, 10, fc="#f9dcdc")
    _box(ax, 4.4, 5.4, 1.4, 0.9, "DataFrame\nbuilder", C_TBL, 10, fc="#dcebda")

    # Storage layer (parallel)
    _box(ax, 2.6, 3.9, 1.4, 0.9, "Vector DB\n(ChromaDB,\nBGE-large)", C_VEC, 10, fc="#f5cccc")
    _box(ax, 4.4, 3.9, 1.4, 0.9, "Table Store\n(Pandas +\nHeaderPath)", C_TBL, 10, fc="#cce5cc")

    _arrow(ax, 3.3, 6.6, 3.3, 6.3, color="#666")
    _arrow(ax, 5.1, 6.6, 5.1, 6.3, color="#666")
    _arrow(ax, 3.3, 5.4, 3.3, 4.8, color=C_VEC, label="embed")
    _arrow(ax, 5.1, 5.4, 5.1, 4.8, color=C_TBL, label="parse")

    # ---- QUERY-TIME AGENT ----
    ax.text(7.4, 7.7, "QUERY TIME (per request)", fontsize=10, color="#666", style="italic")

    # Pipeline stages
    _box(ax, 6.4, 5.7, 2.0, 0.8, "1) Retrieve\n(vector top-k)", C_AGENT, 10, fc="#e3d6f0")
    _box(ax, 6.4, 4.4, 2.0, 0.8, "2) Verify\n(query→table)", C_AGENT, 10, fc="#e3d6f0")
    _box(ax, 6.4, 3.1, 2.0, 0.8, "3) Reconcile\n(rerank/filter)", C_AGENT, 10, fc="#e3d6f0")
    _box(ax, 6.4, 1.8, 2.0, 0.8, "4) Answer\n(local LLM)", C_AGENT, 10, fc="#e3d6f0")
    _box(ax, 6.4, 0.5, 2.0, 0.8, "5) Trace\n(cell grounding)", C_AGENT, 10, fc="#e3d6f0")

    # Query feeds each stage that needs it
    _arrow(ax, 2.1, 4.4, 6.4, 6.1, color=C_QUERY, lw=1.2)
    _arrow(ax, 2.1, 4.4, 6.4, 4.8, color=C_QUERY, lw=1.0, ls="--")

    # Vector DB → Retrieve
    _arrow(ax, 4.0, 4.35, 6.4, 5.95, color=C_VEC, label="hits")

    # Table Store → Verify (key cross-check)
    _arrow(ax, 5.8, 4.35, 6.4, 4.65, color=C_TBL, label="cells")
    # Table Store → Answer (LLM consumes top-1 table)
    _arrow(ax, 5.8, 4.0, 6.4, 2.1, color=C_TBL, lw=1.0, ls="--", label="top-1 table")
    # Table Store → Trace
    _arrow(ax, 5.8, 3.95, 6.4, 0.75, color=C_TBL, lw=1.0, ls="--")

    # Vertical pipeline arrows
    _arrow(ax, 7.4, 5.7, 7.4, 5.25, color=C_AGENT)
    _arrow(ax, 7.4, 4.4, 7.4, 3.95, color=C_AGENT)
    _arrow(ax, 7.4, 3.1, 7.4, 2.65, color=C_AGENT)
    _arrow(ax, 7.4, 1.8, 7.4, 1.35, color=C_AGENT)

    # ---- OUTPUTS ----
    ax.text(11.5, 7.7, "OUTPUTS", fontsize=10, color="#666", style="italic")
    _box(ax, 9.5, 5.7, 3.5, 0.9,
         "Ranked tables\n(R@k, MRR — TARGET / HiTab)", C_OUT, 10, fc="#fde6cc")
    _box(ax, 9.5, 4.0, 3.5, 0.9,
         "Disagreement signal\n(vector-only ⟷ verified)", C_OUT, 10, fc="#fde6cc")
    _box(ax, 9.5, 2.0, 3.5, 0.9,
         "Generated answer\n(Qwen2.5-3B-Instruct, 4-bit)", C_OUT, 10, fc="#fde6cc")
    _box(ax, 9.5, 0.5, 3.5, 0.9,
         "Answer trace → cells\n(grounded_fraction; hallucination flag)", C_OUT, 10, fc="#fde6cc")

    _arrow(ax, 8.4, 6.1, 9.5, 6.1, color=C_OUT)
    _arrow(ax, 8.4, 3.5, 9.5, 4.4, color=C_OUT)
    _arrow(ax, 8.4, 2.2, 9.5, 2.4, color=C_OUT)
    _arrow(ax, 8.4, 0.9, 9.5, 0.9, color=C_OUT)

    # ---- Legend / key result ----
    ax.text(0.3, 1.8,
            "Key idea:\nthe ORIGINAL parsed 2D\ntable is kept alongside the\nvector DB and is consulted\nat query time as a JUDGE\nof the vector hits — not\nas a generator.",
            fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8dc", ec="#bbb"))

    # Footer with headline numbers
    ax.text(7.0, -0.35,
            "HiTab dev (300q, BGE-large): R@1 0.607 → 0.730 (+12.3%p)  |  FeTaQA R@10 0.670 → 0.685 (+1.5%p)  |  TabFact R@10 0.772 → 0.760 (-1.2%p, ablated)",
            fontsize=8.5, color="#333", ha="center", style="italic")

    out = Path("docs")
    out.mkdir(exist_ok=True)
    out_file = out / "sidecar_architecture.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
