#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 1: set-EM@50 decay vs operand-set size (hybrid retriever, n=297).

Publication style (ACL/EMNLP conventions): no in-figure title — the message
lives in the LaTeX caption (suggested caption kept in PAPER_DRAFT.md §5.1c);
serif fonts sized for a single-column (~3.3in) figure; legend box; series
distinguishable in grayscale (marker + linestyle, not color alone); Wilson 95%
CIs (the 9+ bin has n=5 — the CI makes that honest instead of hiding it).

Reads results/operand_collision_multihiertt_n300_scope_slices.json (P4 output);
writes docs/fig1_scope_decay.{png,pdf}.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,          # editable text in the PDF (camera-ready req.)
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
BIN_ORDER = ["2", "3-4", "5-8", "9+"]
K = "@50"
RETRIEVER = "hybrid"
STYLE = {  # color + marker + linestyle: identity never rides on color alone
    "flat": dict(color="#eda100", marker="^", linestyle="--"),
    "S2": dict(color="#1baf7a", marker="s", linestyle="-"),
    "S3": dict(color="#2a78d6", marker="o", linestyle="-"),
}
LABEL = {"flat": "flat", "S2": "S2 (header path)", "S3": "S3 (caption)"}


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

    fig, ax = plt.subplots(figsize=(3.3, 2.3), dpi=300)

    for scheme in ("flat", "S2", "S3"):
        ys = [block[b]["set_recall"][scheme][K] for b in BIN_ORDER]
        los, his = zip(*(wilson(y, n) for y, n in zip(ys, ns)))
        errs = [[y - lo for y, lo in zip(ys, los)],
                [hi - y for y, hi in zip(ys, his)]]
        ax.errorbar(xs, ys, yerr=errs, lw=1.1, ms=3.5, capsize=1.8,
                    elinewidth=0.7, label=LABEL[scheme], clip_on=False,
                    **STYLE[scheme])

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}\n($n$={n})" for b, n in zip(BIN_ORDER, ns)])
    ax.set_xlabel("operand-set size $m$")
    ax.set_ylabel("set-EM@50")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.2, len(BIN_ORDER) - 0.8)
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8,
              borderaxespad=0.2)

    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        out = ROOT / f"docs/fig1_scope_decay.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
