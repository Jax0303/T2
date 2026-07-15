#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 1: set-EM@50 decay vs operand-set size (hybrid retriever, n=297).

The paper's headline shape: completeness decays as the aggregation scope grows,
flat decays fastest, and the S3-flat gap widens monotonically — completeness
failure concentrates exactly where aggregation needs completeness most.

Reads results/operand_collision_multihiertt_n300_scope_slices.json (P4 output);
draws per-scheme decay curves with Wilson 95% CIs (the 9+ bin has n=5 — the CI
makes that honest instead of hiding it). Writes docs/fig1_scope_decay.{png,pdf}.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BIN_ORDER = ["2", "3-4", "5-8", "9+"]
K = "@50"
RETRIEVER = "hybrid"
# validated reference palette, fixed categorical order (slot1..3)
COLOR = {"S3": "#2a78d6", "S2": "#1baf7a", "flat": "#eda100"}
LABEL = {"S3": "S3 caption", "S2": "S2 header-path", "flat": "flat (naive cells)"}
INK, INK2 = "#0b0b0b", "#52514e"


def wilson(p: float, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 1.0
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    src = ROOT / "results/operand_collision_multihiertt_n300_scope_slices.json"
    block = json.loads(src.read_text())["by_retriever"][RETRIEVER]
    xs = list(range(len(BIN_ORDER)))
    ns = [block[b]["n_queries"] for b in BIN_ORDER]

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    end_y = {s: block[BIN_ORDER[-1]]["set_recall"][s][K]
             for s in ("flat", "S2", "S3")}
    # dodge coincident end-of-line labels apart (min separation in data units)
    label_y = dict(end_y)
    order = sorted(label_y, key=label_y.get)
    for a, b in zip(order, order[1:]):
        if label_y[b] - label_y[a] < 0.06:
            label_y[b] = label_y[a] + 0.06

    for scheme in ("flat", "S2", "S3"):
        ys = [block[b]["set_recall"][scheme][K] for b in BIN_ORDER]
        los, his = zip(*(wilson(y, n) for y, n in zip(ys, ns)))
        errs = [[y - lo for y, lo in zip(ys, los)],
                [hi - y for y, hi in zip(ys, his)]]
        ax.errorbar(xs, ys, yerr=errs, color=COLOR[scheme], lw=2,
                    marker="o", ms=5, capsize=2.5, elinewidth=0.9,
                    alpha=0.95, linestyle="--" if scheme == "flat" else "-",
                    zorder=3 if scheme == "flat" else 4)
        ax.annotate(LABEL[scheme], (xs[-1] + 0.12, label_y[scheme]),
                    color=COLOR[scheme], fontsize=8.5, va="center",
                    fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}*" if block[b]["flip_test"]["S3"][K]
                        ["p_two_sided"] < 0.05 else b for b in BIN_ORDER],
                       fontsize=9, color=INK)
    for x, n in zip(xs, ns):
        ax.annotate(f"n={n}", (x, -0.145), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("operand-set size (gold cells per aggregation)", fontsize=9,
                  color=INK, labelpad=14)
    ax.set_ylabel(f"set-EM{K} (all operands retrieved)", fontsize=9, color=INK)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.25, len(BIN_ORDER) + 0.85)
    ax.grid(axis="y", color="#e8e8e6", lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_title("Completeness decays with aggregation scope — flat fastest\n"
                 "(MultiHiertt, hybrid retriever, 297 queries; "
                 "* flat→S3 flip test p<.05; 95% Wilson CI)",
                 fontsize=9, color=INK, loc="left", pad=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = ROOT / f"docs/fig1_scope_decay.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
