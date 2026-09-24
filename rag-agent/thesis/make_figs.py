"""논문 그림. 그림 3-1 은 방법 개요(데이터 없음), 그림 5-x 는 results/ 파일에서 읽는다.
  .venv/bin/python thesis/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "fig"
BLUE, INK, MUTED, FILL = "#2a78d6", "#0b0b0b", "#52514e", "#f0efec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 2.6), dpi=200)
    ax.set_xlim(0, 102); ax.set_ylim(0, 32); ax.axis("off")
    steps = [("Hierarchical\ntable", "row / column\nheader paths", False),
             ("Cell\nsentence", "label + row path\n+ column path\n+ value", True),
             ("Index", "bge-base-en-v1.5\n+ BM25\n(no training)", False),
             ("Hybrid\nsearch", "0.7 dense\n+ 0.3 BM25", False),
             ("Top-20\ncells", "budget =\n20 cells", False),
             ("Local\nreader", "Qwen 4-bit", False)]
    w, gap, y = 14.2, 2.8, 8
    for k, (head, sub, key) in enumerate(steps):
        x = 1 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, 18, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#e3eefb" if key else FILL, ec=BLUE if key else MUTED, lw=1.4 if key else 0.8))
        ax.text(x + w / 2, y + 13.2, head, ha="center", va="center", fontsize=8.2, weight="bold", color=INK)
        ax.text(x + w / 2, y + 5, sub, ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.3)
        if k < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.3, y + 9), xytext=(x + w + 0.4, y + 9),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.text(50, 2.5, "Question  →  search only the cells of the paired table (HiTab) or document (MultiHiertt)",
            ha="center", fontsize=7.4, color=MUTED)
    fig.savefig(OUT / "fig3_1_pipeline.png", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    pipeline()
