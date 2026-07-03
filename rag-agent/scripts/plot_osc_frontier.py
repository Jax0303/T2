#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure: total-row injection vs similarity retrieval — frontier + strict tests.

Three panels:
  A. OSC↔cells frontier (HiTab dev): plain curves vs +injection curves over the
     budget sweep — the augmented curve dominating = Pareto improvement.
  B. STRICT cell-matched paired test (dev): aug@10 vs plain@k' given the SAME
     per-query cell budget; the win is structural, not budget.
  C. Held-out confirmation (HiTab test, frozen config): same strict test.

Sources: results/osc_total_augment.json (dev, bge-reranker resolver) and
results/osc_total_augment_TESTSPLIT.json (frozen test run).

Run: PYTHONPATH=. python scripts/plot_osc_frontier.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DEV = "results/osc_total_augment.json"
SRC_TEST = "results/osc_total_augment_TESTSPLIT.json"
OUT = "docs/fig_osc_frontier.png"
COLORS = {"bm25": "#d1495b", "dense": "#2e7d32", "hybrid": "#1565c0"}
LABELS = {"bm25": "BM25", "dense": "dense", "hybrid": "hybrid"}
METHODS = ("bm25", "dense", "hybrid")


def frontier_panel(ax, d):
    for m in METHODS:
        ks = d["methods"][m]
        order = sorted(ks.items(), key=lambda kv: kv[1]["cells_plain"])
        cp = [r["cells_plain"] for _, r in order]
        op = [r["osc_plain"] for _, r in order]
        ca = [r["cells_aug"] for _, r in order]
        oa = [r["osc_aug"] for _, r in order]
        c = COLORS[m]
        ax.plot(cp, op, "--o", color=c, alpha=0.55, mfc="white", lw=1.5, ms=5,
                label=f"{LABELS[m]} (plain)")
        ax.plot(ca, oa, "-s", color=c, lw=2.2, ms=6,
                label=f"{LABELS[m]} +inject")
    inj = d.get("mean_total_cells_injected", "?")
    ax.set_xlabel("retrieved cells (budget)", fontsize=11)
    ax.set_ylabel("OSC (operand-set completeness)", fontsize=11)
    ax.set_title(f"A. OSC↔cells frontier (dev, n=161)\ninjection ≈ +{inj} cells",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7.5, ncol=1, loc="lower right", framealpha=0.9)


def strict_panel(ax, d, title, n):
    cm = d["cell_matched_test"]["methods"]
    x = np.arange(len(METHODS))
    w = 0.36
    plain = [cm[m]["osc_plain_matched"] for m in METHODS]
    aug = [cm[m]["osc_aug"] for m in METHODS]
    ax.bar(x - w / 2, plain, w, color=[COLORS[m] for m in METHODS], alpha=0.35,
           label="plain @ matched cells", edgecolor="white")
    ax.bar(x + w / 2, aug, w, color=[COLORS[m] for m in METHODS],
           label="+injection @ k=10", edgecolor="white")
    for i, m in enumerate(METHODS):
        p = cm[m]["mcnemar_p"]
        star = "***" if p < 0.001 else ("**" if p < 0.01 else
                                        ("*" if p < 0.05 else "n.s."))
        top = max(plain[i], aug[i])
        ax.text(x[i], top + 0.015, f"{star}\np={p:.3g}", ha="center",
                fontsize=8)
        ax.text(x[i] + w / 2, aug[i] - 0.045, f"{aug[i]:.3f}", ha="center",
                fontsize=7.5, color="white", fontweight="bold")
        ax.text(x[i] - w / 2, plain[i] - 0.045, f"{plain[i]:.3f}", ha="center",
                fontsize=7.5, color="#333")
    ax.set_xticks(x, [LABELS[m] for m in METHODS], fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.set_title(f"{title}\nstrict equal-cell budget (n={n})", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


def main() -> int:
    dev = json.load(open(SRC_DEV))
    test = json.load(open(SRC_TEST))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    frontier_panel(axes[0], dev)
    strict_panel(axes[1], dev, "B. Dev (tuned)", dev["population"]["n"])
    strict_panel(axes[2], test, "C. Test (frozen, held-out)",
                 test["population"]["n"])

    fig.suptitle("Total-row injection beats BM25/dense/hybrid on OSC — "
                 "Pareto-dominant (A), under a strict equal-cell budget (B), "
                 "and on the frozen held-out split (C)",
                 fontsize=12, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
