#!/usr/bin/env python3
"""Publication-style figures for the preprocessing-ladder retrieval result.

Fig 1: grouped bar of R@1/R@5/R@10 across C0-C3 with 95% bootstrap CIs.
Fig 2: recall vs preprocessing condition (ladder), one line per k.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, B = 42, 10000
CONDS = ["C0", "C1", "C2", "C3"]
COND_SUB = {"C0": "raw", "C1": "+meta", "C2": "+schema", "C3": "+synthQ"}
KS = [1, 5, 10]

d = json.load(open("results/prep/owt_bm25_n1000.json"))
pq = d["per_query"]
ranks = {c: np.array([r if r is not None else 10**9 for r in pq[c]], dtype=float)
         for c in CONDS}
n = len(ranks["C0"])

def recall_at(arr, k):
    return float(np.mean(arr <= k))

def boot_ci(arr, k):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    samp = (arr[idx] <= k).mean(axis=1)
    return np.percentile(samp, 2.5), np.percentile(samp, 97.5)

point = {c: {k: recall_at(ranks[c], k) for k in KS} for c in CONDS}
ci = {c: {k: boot_ci(ranks[c], k) for k in KS} for c in CONDS}

# ---- publication aesthetics ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "figure.dpi": 200,
})
# muted, print-friendly palette (one shade per k)
PAL = {1: "#1b4965", 5: "#5fa8d3", 10: "#bee3db"}

# ===================== Fig 1: grouped bar =====================
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(len(CONDS)); w = 0.26
for j, k in enumerate(KS):
    vals = [point[c][k] for c in CONDS]
    lo = [point[c][k] - ci[c][k][0] for c in CONDS]
    hi = [ci[c][k][1] - point[c][k] for c in CONDS]
    bars = ax.bar(x + (j-1)*w, vals, w, yerr=[lo, hi], capsize=3,
                  color=PAL[k], edgecolor="black", linewidth=0.6,
                  error_kw=dict(elinewidth=1.0, ecolor="#333"),
                  label=f"R@{k}")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n{COND_SUB[c]}" for c in CONDS])
ax.set_ylabel("Recall")
ax.set_xlabel("Preprocessing condition (cumulative)")
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.02), columnspacing=1.2)
ax.set_title("Table preprocessing lifts retrieval (OpenWikiTable, BM25, n=1000)",
             fontsize=11.5, pad=24)
fig.tight_layout()
fig.savefig("/home/user/T2/rag-agent/docs/fig_bm25_bars.png", bbox_inches="tight",
            facecolor="white")
print("fig1 saved")

# ===================== Fig 2: recall ladder =====================
fig, ax = plt.subplots(figsize=(6.4, 4.2))
MARK = {1: "o", 5: "s", 10: "^"}
for k in KS:
    vals = [point[c][k] for c in CONDS]
    los = [ci[c][k][0] for c in CONDS]
    his = [ci[c][k][1] for c in CONDS]
    ax.plot(x, vals, marker=MARK[k], color=PAL[k], linewidth=2, markersize=7,
            markeredgecolor="black", markeredgewidth=0.6, label=f"R@{k}", zorder=3)
    ax.fill_between(x, los, his, color=PAL[k], alpha=0.15, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n{COND_SUB[c]}" for c in CONDS])
ax.set_ylabel("Recall")
ax.set_xlabel("Preprocessing condition (cumulative)")
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower right")
# annotate the dominant jump C0->C1 on R@1
dy = point["C1"][1] - point["C0"][1]
ax.annotate(f"+{dy:.2f}", xy=(0.5, (point['C0'][1]+point['C1'][1])/2),
            fontsize=11, color="#b00", ha="center",
            fontweight="bold")
ax.set_title("Most of the gain comes from metadata (C0→C1)",
             fontsize=11.5, pad=10)
fig.tight_layout()
fig.savefig("/home/user/T2/rag-agent/docs/fig_bm25_ladder.png", bbox_inches="tight",
            facecolor="white")
print("fig2 saved")

# print the numbers used (for slide text)
print("\n--- numbers ---")
for c in CONDS:
    s = " ".join(f"R@{k}={point[c][k]:.3f}[{ci[c][k][0]:.3f},{ci[c][k][1]:.3f}]" for k in KS)
    print(c, s)
