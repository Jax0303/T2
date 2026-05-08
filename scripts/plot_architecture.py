# SPDX-License-Identifier: MIT
"""Generate experiment architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis("off")
fig.patch.set_facecolor("white")

# Colors
C_DATA = "#E3F2FD"
C_SER = "#FFF3E0"
C_METRIC = "#E8F5E9"
C_PROBE = "#F3E5F5"
C_MODEL = "#FCE4EC"
C_RESULT = "#FFFDE7"
C_ARROW = "#37474F"
C_TITLE = "#1A237E"
C_SEC3 = "#E65100"
C_SEC4 = "#4A148C"


def box(x, y, w, h, text, color, fontsize=10, bold=False, border="#455A64", lw=1.5, alpha=1.0):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=border, linewidth=lw, alpha=alpha,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, fontweight=weight, color="black", linespacing=1.4,
    )


def arrow(x1, y1, x2, y2, connectionstyle="arc3,rad=0"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=2, connectionstyle=connectionstyle),
    )


def section_label(x, y, text, color):
    ax.text(
        x, y, text, fontsize=14, fontweight="bold", color=color, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, lw=2),
    )


def small_box(x, y, w, h, title, desc, face, edge):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=face, edgecolor=edge, linewidth=1,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=8, fontweight="bold")
    if desc:
        ax.text(x + w / 2, y + h * 0.28, desc, ha="center", va="center", fontsize=6.5, color="#555")


# =====================================================================
# Title
# =====================================================================
ax.text(
    11, 15.4, "Table-RAG Structural Audit: Experiment Architecture",
    ha="center", va="center", fontsize=18, fontweight="bold", color=C_TITLE,
)

# Section labels
section_label(5.5, 14.4, "§3  Serialization Damage Diagnosis", C_SEC3)
section_label(16.5, 14.4, "§4  Embedder Layer-wise Probing", C_SEC4)

# Divider
ax.plot([11, 11], [0.5, 14.0], color="#BDBDBD", lw=2, ls="--", alpha=0.6)

# =====================================================================
# LEFT SIDE (§3)
# =====================================================================

# HiTab
box(3.5, 12.5, 4, 1, "HiTab Dataset\n3,597 tables (100 sampled)", C_DATA, 11, True)
arrow(5.5, 12.5, 5.5, 11.9)

# Table object
box(3.5, 10.8, 4, 1.1, "Table Object\ncells · header trees\nmerged_cells · metadata", C_DATA, 9)
arrow(5.5, 10.8, 5.5, 10.2)

# Serializers container
box(0.5, 8.8, 10, 1.4, "", C_SER, 9, border=C_SEC3, lw=2)
ax.text(5.5, 9.95, "5 Serializers", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SEC3)

ser_data = [
    ("HTML", "#BBDEFB", "lossless\nspans"),
    ("Markdown", "#FFE0B2", "lossy\nflat"),
    ("CSV", "#C8E6C9", "lossy\nflat"),
    ("JSON-tree", "#E1BEE7", "fully\nlossless"),
    ("OTSL", "#FFCDD2", "span\ntokens"),
]
for i, (name, sc, sl) in enumerate(ser_data):
    small_box(0.8 + i * 1.95, 8.95, 1.7, 0.85, name, sl, sc, "#78909C")

# Round-trip label
arrow(5.5, 8.8, 5.5, 8.15)
ax.text(5.5, 8.47, "serialize → parse (round-trip)", ha="center", va="center", fontsize=8, color="#555")

# Recovered table
box(3.5, 7.1, 4, 0.8, "Recovered Table", C_SER, 10, alpha=0.8)
arrow(5.5, 7.1, 5.5, 6.5)

# Metrics container
box(0.5, 4.5, 10, 2.0, "", C_METRIC, 9, border=C_SEC3, lw=2)
ax.text(5.5, 6.25, "4 Structural Metrics", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SEC3)

met_data = [
    ("TEDS", "Tree-Edit Distance\nSimilarity (zss)"),
    ("Header Path\nAccuracy", "Ancestor chain\nmatch rate"),
    ("Cell Coord\nPreservation", "(r,c)→value\nmatch rate"),
    ("Merged Cell\nRecovery", "rowspan/colspan\nrecovery rate"),
]
for i, (mname, mdesc) in enumerate(met_data):
    small_box(0.8 + i * 2.45, 4.65, 2.2, 1.35, mname, mdesc, "#C8E6C9", "#66BB6A")

# Results
arrow(5.5, 4.5, 5.5, 3.9)
box(1.5, 2.5, 8, 1.4, "", C_RESULT, 9, border="#F9A825", lw=2)
ax.text(5.5, 3.55, "Results: 5×4×100 Grid", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SEC3)
ax.text(5.5, 2.95, "Paired bootstrap CI (n=1000)\nCSV + Markdown table + Box plot PDF", ha="center", va="center", fontsize=9, color="#555")

# =====================================================================
# RIGHT SIDE (§4)
# =====================================================================

# Shared data arrow
arrow(7.5, 13.0, 12.2, 13.0, connectionstyle="arc3,rad=-0.15")
ax.text(9.8, 13.35, "same 100\ntables", ha="center", va="center", fontsize=7.5, color="#888")

# Cell texts
box(12.2, 12.5, 4.2, 1, "Cell Texts\n(unique values extracted)", C_DATA, 10)
arrow(14.3, 12.5, 14.3, 11.8)

# Models container
box(12, 10.5, 5, 1.3, "", C_MODEL, 9, border=C_SEC4, lw=2)
ax.text(14.5, 11.5, "Sentence-Transformer Models", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SEC4)
for i, mname in enumerate(["BGE-small-en-v1.5", "E5-small-v2"]):
    small_box(12.3 + i * 2.5, 10.65, 2.2, 0.7, mname, "", "#F8BBD0", "#E91E63")

arrow(14.5, 10.5, 14.5, 9.9)

# Hidden states
box(12, 8.8, 5, 1.1, "Hidden States\n13 layers × hidden_dim\n(embedding + 12 transformer)", C_MODEL, 9, alpha=0.85)
arrow(14.5, 8.8, 14.5, 8.2)

# Probe tasks container
box(11.5, 6.2, 6, 2.0, "", C_PROBE, 9, border=C_SEC4, lw=2)
ax.text(14.5, 7.95, "3 Probe Tasks", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SEC4)

task_data = [
    ("Parent Header", "Hash-bucket of\nancestor header"),
    ("Cell Coordinate", "(row, col)\nbucket prediction"),
    ("Same Row", "Binary: same\nrow or not?"),
]
for i, (tname, tdesc) in enumerate(task_data):
    small_box(11.8 + i * 1.9, 6.35, 1.7, 1.35, tname, tdesc, "#E1BEE7", "#AB47BC")

arrow(14.5, 6.2, 14.5, 5.6)

# Classifiers container
box(12, 4.5, 5, 1.1, "", C_PROBE, 9, border=C_SEC4, lw=2)
ax.text(14.5, 5.35, "Probe Classifiers", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SEC4)
for i, cname in enumerate(["Linear Probe", "MLP (2-layer)"]):
    small_box(12.3 + i * 2.5, 4.6, 2.2, 0.6, cname, "", "#D1C4E9", "#7E57C2")

# Selectivity note
ax.text(
    18.2, 4.9, "+ Selectivity\n  Control\n  (Hewitt &\n   Liang 2019)",
    ha="left", va="center", fontsize=7.5, color=C_SEC4, style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor=C_SEC4, lw=1, ls="--"),
)

arrow(14.5, 4.5, 14.5, 3.9)

# Results
box(12, 2.5, 5, 1.4, "", C_RESULT, 9, border="#F9A825", lw=2)
ax.text(14.5, 3.55, "Results: 2×13×3×2 Grid", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SEC4)
ax.text(14.5, 2.95, "Accuracy + Selectivity per layer\nTenney-2019 style layer curves", ha="center", va="center", fontsize=9, color="#555")

# =====================================================================
# Bottom: constraints
# =====================================================================
box(3, 0.8, 16, 1.2, "", "#ECEFF1", 9, border="#90A4AE", lw=1.5)
ax.text(11, 1.65, "Constraints & Reproducibility", ha="center", va="center", fontsize=11, fontweight="bold", color="#37474F")
ax.text(
    11, 1.1,
    "CPU-only  ·  Python 3.11 + uv  ·  Hydra configs  ·  seed=42  ·  HiTab ground-truth only  ·  53 unit tests",
    ha="center", va="center", fontsize=9, color="#546E7F",
)

fig.savefig("results/figures/architecture_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved architecture_diagram.png")
