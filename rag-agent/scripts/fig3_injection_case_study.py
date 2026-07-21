#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 3 (SS5.10 case study): total-row injection on HiTab, post-audit data.

Replaces the deleted docs/fig_osc_frontier.png, which was drawn from PRE-audit
gold (results committed 2026-07-03; the 2026-07-07 audit changed operand gold
and row headers, and the frozen-test split was never re-run). This figure uses
ONLY the post-audit canonical file results/osc_total_augment_resolver.json
(re-measured 2026-07-08, eff4510) — hence dev only, no test panel.

Panels: (a) OSC vs retrieved-cell budget (plain dashed vs +inject solid, per
retriever) — injection is Pareto-dominant above the starvation regime;
(b) same-depth comparison at k=10 (the SS5.10 headline table as bars).
Publication style: no in-figure title (caption lives in PAPER_DRAFT.md SS5.10),
serif, legends, single-column width, Type-42 fonts.

Writes docs/fig3_injection_case_study.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
COLOR = {"bm25": "#eda100", "dense": "#1baf7a", "hybrid": "#2a78d6"}
MARKER = {"bm25": "^", "dense": "s", "hybrid": "o"}


def main() -> int:
    d = json.loads(
        (ROOT / "results/osc_total_augment_resolver.json").read_text())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(3.3, 1.95), dpi=300,
        gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) OSC vs mean retrieved cells, plain vs +inject
    for ret, per_k in d["methods"].items():
        ks = sorted(per_k, key=lambda s: int(s.lstrip("@")))
        for aug, ls in (("plain", "--"), ("aug", "-")):
            xs = [per_k[k][f"cells_{aug}"] for k in ks]
            ys = [per_k[k][f"osc_{aug}"] for k in ks]
            ax1.plot(xs, ys, ls, color=COLOR[ret], marker=MARKER[ret],
                     ms=2.6, lw=0.9, mew=0.5,
                     alpha=0.55 if aug == "plain" else 1.0)
    ax1.set_xlabel("retrieved cells (budget)")
    ax1.set_ylabel("OSC")
    ax1.set_ylim(0.55, 1.02)
    ax1.grid(color="#dddddd", lw=0.5)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    scheme_handles = [Line2D([], [], color=COLOR[r], marker=MARKER[r],
                             ms=2.6, lw=0.9, label=r)
                      for r in ("bm25", "dense", "hybrid")]
    cond_handles = [Line2D([], [], color="#555555", ls="--", lw=0.9,
                           alpha=0.55, label="plain"),
                    Line2D([], [], color="#555555", ls="-", lw=0.9,
                           label="+inject")]
    ax1.legend(handles=scheme_handles + cond_handles, frameon=False,
               loc="lower right", handlelength=1.7, borderaxespad=0.1,
               labelspacing=0.25)

    # (b) same-depth k=10 bars
    sd = d["same_depth_test"]["methods"]
    for gi, ret in enumerate(("bm25", "dense", "hybrid")):
        for ci, cond in enumerate(("osc_plain", "osc_aug")):
            x = gi * 2.4 + ci
            y = sd[ret][cond]
            ax2.bar(x, y, width=0.9, color=COLOR[ret],
                    alpha=0.45 if ci == 0 else 1.0,
                    hatch="///" if ci == 0 else None,
                    edgecolor="white", linewidth=0.6, zorder=3)
            ax2.annotate(f"{y:.2f}".lstrip("0"), (x, y + 0.008),
                         ha="center", fontsize=5.8)
    ax2.set_xticks([0.5, 2.9, 5.3])
    ax2.set_xticklabels(["bm25", "dense", "hybrid"])
    ax2.set_ylim(0.55, 1.02)
    ax2.set_title("same depth, $k{=}10$", fontsize=7.5, pad=2)
    ax2.grid(axis="y", color="#dddddd", lw=0.5)
    ax2.set_axisbelow(True)
    ax2.set_yticklabels([])
    ax2.tick_params(axis="both", length=0)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout(pad=0.4, w_pad=0.8)
    for ext in ("png", "pdf"):
        out = ROOT / f"docs/fig3_injection_case_study.{ext}"
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
