#!/usr/bin/env python3
"""Render a prep-retrieval result JSON as a publication-style (booktabs) table.

Usage:
  python scripts/draw_results_table.py \
      --json results/prep/owt_bm25_n1000.json \
      --conditions C0,C1,C2,C3 \
      --caption "OpenWikiTable (24,680 tables), BM25, n=1000, seed=42." \
      --out docs/table_bm25_results.png
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESC = {
    "C0": "raw table only",
    "C1": "+ title / section / caption",
    "C2": "+ column schema",
    "C2h": "+ header-path schema",
    "C3": "+ synthetic questions",
}
# conditions to visually flag as structure-aware (header-path)
HILITE = {"C2h"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--conditions", default="C0,C1,C2,C3")
    p.add_argument("--caption", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = Path(__file__).resolve().parents[1]
    d = json.load(open(base / args.json))
    C = d["conditions"]
    order = [c.strip() for c in args.conditions.split(",")]
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    colmax = {m: max(C[c][m] for c in order) for m in metrics}

    header = ["Cond.", "Preprocessing", "R@1", "R@5", "R@10", "MRR"]
    xs = [0.0, 0.085, 0.56, 0.69, 0.82, 0.95]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    h = 1.1 + 0.42 * len(order)
    fig, ax = plt.subplots(figsize=(9.4, h))
    ax.axis("off")

    y0 = 0.84
    dy = 0.62 / len(order) if len(order) <= 4 else 0.155
    dy = 0.155

    for j, hh in enumerate(header):
        ha = "left" if j <= 1 else "center"
        ax.text(xs[j], y0, hh, fontsize=12.5, fontweight="bold", ha=ha, va="center")
    ax.plot([0, 1.0], [y0 + 0.09, y0 + 0.09], color="black", lw=1.6)
    ax.plot([0, 1.0], [y0 - 0.06, y0 - 0.06], color="black", lw=1.0)

    for i, c in enumerate(order):
        yy = y0 - 0.06 - dy * (i + 1) + 0.04
        hil = c in HILITE
        rowcells = [c, DESC.get(c, c)] + [f"{C[c][m]:.3f}" for m in metrics]
        for j, txt in enumerate(rowcells):
            ha = "left" if j <= 1 else "center"
            m = header[j] if j >= 2 else None
            best = (m is not None and abs(C[c][m] - colmax[m]) < 1e-9)
            color = "#b00000" if hil else ("#0a4d8c" if best else "black")
            weight = "bold" if (best or hil) else "normal"
            ax.text(xs[j], yy, txt, fontsize=12, ha=ha, va="center",
                    fontweight=weight, color=color)

    ybot = y0 - 0.06 - dy * len(order) + 0.01
    ax.plot([0, 1.0], [ybot, ybot], color="black", lw=1.6)
    cap = args.caption + ("  Bold blue = best per column; red = structure-aware (header-path).")
    ax.text(0.0, ybot - 0.10, cap, fontsize=8.5, color="#444", ha="left", va="top")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(ybot - 0.22, 1.0)

    fig.savefig(base / args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", args.out)
    for c in order:
        print(c, {m: round(C[c][m], 3) for m in metrics})


if __name__ == "__main__":
    main()
