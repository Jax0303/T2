#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plot the OSC↔cells frontier: plain similarity retrieval vs +total-row injection.

Reads results/osc_total_augment_resolver.json (resolver-targeted injection, the cheap
~6-cell patch) and draws, per retriever (BM25/dense/hybrid), the plain curve vs the
+injection curve over the budget sweep. The augmented curve dominating shows the
structural patch is a Pareto improvement (higher OSC at the same cell budget).

Run: PYTHONPATH=. python scripts/plot_osc_frontier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results/osc_total_augment_resolver.json"
OUT = "docs/fig_osc_frontier.png"
COLORS = {"bm25": "#d1495b", "dense": "#2e7d32", "hybrid": "#1565c0"}
LABELS = {"bm25": "BM25", "dense": "dense", "hybrid": "hybrid"}


def main() -> int:
    d = json.load(open(SRC))
    methods = d["methods"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for m, ks in methods.items():
        order = sorted(ks.items(), key=lambda kv: kv[1]["cells_plain"])
        cp = [r["cells_plain"] for _, r in order]
        op = [r["osc_plain"] for _, r in order]
        ca = [r["cells_aug"] for _, r in order]
        oa = [r["osc_aug"] for _, r in order]
        c = COLORS[m]
        ax.plot(cp, op, "--o", color=c, alpha=0.55, mfc="white", lw=1.6, ms=6,
                label=f"{LABELS[m]} (plain)")
        ax.plot(ca, oa, "-s", color=c, lw=2.4, ms=7, label=f"{LABELS[m]} + total-row inject")

    ax.set_xlabel("retrieved cells (budget)", fontsize=12)
    ax.set_ylabel("OSC (operand-set completeness)", fontsize=12)
    ax.set_title("Total-row injection Pareto-dominates similarity retrieval\n"
                 "HiTab dev, arithmetic m≥2 (n=161); injection ≈ +6 cells",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.02)
    # legend grouped: solid=augmented, dashed=plain
    ax.legend(fontsize=9, ncol=1, loc="lower right", framealpha=0.9)
    ann = (f"dense full-set completeness plateaus (ceiling);\n"
           f"76% of its failures = unreached total rows")
    ax.annotate(ann, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8.5,
                style="italic", color="#444")
    fig.tight_layout()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
