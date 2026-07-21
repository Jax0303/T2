#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 2: the strong-reranker 2x2 — reranking cannot buy completeness.

{flat, S3} x {hybrid pool order, bge-reranker-large} on the SAME top-100 pools
(n=297), at k=10 and k=50; dashed line = each scheme's pool ceiling@100 (the
score a PERFECT reranker over that pool would get). The exhibit: reranking
hurts @10 in both schemes, and flat's ceiling (.566) sits below S3's actual
@50 (.593) — the gain is candidate generation, not ranking.

Publication style (ACL/EMNLP conventions): no in-figure title (message goes in
the LaTeX caption, kept in PAPER_DRAFT.md §5.1c); serif fonts; legend box;
condition encoded by fill+hatch (grayscale-safe), scheme by color+position.

Reads results/operand_collision_rerank_n300.json;
writes docs/fig2_reranker_2x2.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.6,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
COLOR = {"flat": "#eda100", "S3": "#2a78d6"}


def main() -> int:
    d = json.loads(
        (ROOT / "results/operand_collision_rerank_n300.json").read_text())
    by = d["by_scheme"]

    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), dpi=300, sharey=True)

    for ax, k in zip(axes, (10, 50)):
        xs, ticks = [], []
        for gi, scheme in enumerate(("flat", "S3")):
            for ci, cond in enumerate(("hybrid_pool100", "rerank_pool100")):
                x = gi * 2.4 + ci
                y = by[scheme][cond][f"set_recall@{k}"]
                ax.bar(x, y, width=0.9, color=COLOR[scheme],
                       alpha=1.0 if ci == 0 else 0.45,
                       hatch=None if ci == 0 else "///",
                       edgecolor="white", linewidth=0.6, zorder=3)
                ax.annotate(f"{y:.2f}".lstrip("0"), (x, y + 0.015),
                            ha="center", fontsize=6.3)
                xs.append(x)
            ticks.append(scheme)
            ceil = by[scheme]["pool_ceiling@100"]
            ax.hlines(ceil, gi * 2.4 - 0.6, gi * 2.4 + 1.6,
                      color=COLOR[scheme], linestyle=(0, (2, 1.4)), lw=0.9,
                      zorder=4)
        ax.set_xticks([0.5, 2.9])
        ax.set_xticklabels(ticks)
        ax.set_title(f"$k={k}$", fontsize=8, pad=3)
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(0, 0.75)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("set-EM@$k$")

    handles = [
        Patch(facecolor="#9a9a9a", edgecolor="white", label="pool order"),
        Patch(facecolor="#9a9a9a", alpha=0.45, hatch="///",
              edgecolor="white", label="reranked"),
        Line2D([], [], color="#555555", linestyle=(0, (2, 1.4)), lw=0.9,
               label="pool ceiling@100"),
    ]
    fig.legend(handles=handles, ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.55, -0.06),
               handlelength=1.6, columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        out = ROOT / f"docs/fig2_reranker_2x2.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[out] {out}")
    return 0


def _no_args() -> None:
    """This script takes no options. Without a parser, argparse-style flags are
    silently ignored and the full experiment runs anyway — which is how a bare
    ``--help`` sweep silently regenerated committed artifacts."""
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _no_args()
    raise SystemExit(main())
