#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 2: the strong-reranker 2x2 — reranking cannot buy completeness.

{flat, S3} x {hybrid pool order, bge-reranker-large} on the SAME top-100 pools
(n=297). Two panels (set-EM@10, set-EM@50); dashed lines = each scheme's
pool ceiling@100, the score a PERFECT reranker over that pool would get.
The exhibit: reranking *hurts* @10 in both schemes, and flat's ceiling (.566)
sits below S3's actual @50 (.593) — the gain is candidate generation.

Reads results/operand_collision_rerank_n300.json;
writes docs/fig2_reranker_2x2.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COLOR = {"S3": "#2a78d6", "flat": "#eda100"}
INK, INK2 = "#0b0b0b", "#52514e"


def main() -> int:
    d = json.loads(
        (ROOT / "results/operand_collision_rerank_n300.json").read_text())
    by = d["by_scheme"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, k in zip(axes, (10, 50)):
        ax.set_facecolor("white")
        xs, ticks = [], []
        for gi, scheme in enumerate(("flat", "S3")):
            for ci, (cond, cond_label) in enumerate(
                    (("hybrid_pool100", "pool\norder"),
                     ("rerank_pool100", "re-\nranked"))):
                x = gi * 2.9 + ci * 1.15
                y = by[scheme][cond][f"set_recall@{k}"]
                ax.bar(x, y, width=0.92, color=COLOR[scheme],
                       alpha=1.0 if ci == 0 else 0.55,
                       hatch=None if ci == 0 else "//",
                       edgecolor="white", linewidth=1, zorder=3)
                ax.annotate(f"{y:.3f}", (x, y + 0.012), ha="center",
                            fontsize=7.5, color=INK)
                xs.append(x)
                ticks.append(f"{scheme}\n{cond_label}")
            ceil = by[scheme]["pool_ceiling@100"]
            ax.hlines(ceil, gi * 2.9 - 0.65, gi * 2.9 + 1.8,
                      color=COLOR[scheme], linestyle=":", lw=1.4, zorder=4)
            if k == 50:
                ax.annotate(f"perfect-reranker\nceiling {ceil:.3f}",
                            (gi * 2.9 + 0.58, ceil + 0.015), ha="center",
                            fontsize=6.8, color=COLOR[scheme])
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=7.5, color=INK)
        ax.set_title(f"set-EM@{k}", fontsize=9, color=INK)
        ax.grid(axis="y", color="#e8e8e6", lw=0.7, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c4")
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_ylim(0, 0.78)
    axes[0].set_ylabel("set-EM (all operands retrieved)", fontsize=9,
                       color=INK)
    fig.suptitle("A strong reranker cannot buy completeness — "
                 "flat's perfect-reranker ceiling (.566) < S3's actual @50 (.593)",
                 fontsize=9, color=INK, x=0.02, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        out = ROOT / f"docs/fig2_reranker_2x2.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
