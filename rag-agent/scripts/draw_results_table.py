#!/usr/bin/env python3
"""Render the BM25 preprocessing results as a publication-style (booktabs) table."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parents[1]
d = json.load(open(base / "results/prep/owt_bm25_n1000.json"))
C = d["conditions"]
DESC = {"C0": "raw table only", "C1": "+ title / section / caption",
        "C2": "+ column schema", "C3": "+ synthetic questions"}
rows_order = ["C0", "C1", "C2", "C3"]
metrics = ["R@1", "R@5", "R@10", "MRR"]

# build cell text; bold the column-max
colmax = {m: max(C[c][m] for c in rows_order) for m in metrics}
table_rows = []
for c in rows_order:
    cells = [c, DESC[c]]
    for m in metrics:
        v = C[c][m]
        cells.append(f"{v:.3f}")
    table_rows.append(cells)

header = ["Cond.", "Preprocessing", "R@1", "R@5", "R@10", "MRR"]

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.4, 2.5))
ax.axis("off")

ncol = len(header)
# column x-positions
xs = [0.0, 0.085, 0.56, 0.69, 0.82, 0.95]
y0 = 0.82
dy = 0.155

# header
for j, h in enumerate(header):
    ha = "left" if j <= 1 else "center"
    ax.text(xs[j], y0, h, fontsize=12.5, fontweight="bold", ha=ha, va="center")
# rules (booktabs)
ax.plot([0, 1.0], [y0+0.09, y0+0.09], color="black", lw=1.6)   # top
ax.plot([0, 1.0], [y0-0.06, y0-0.06], color="black", lw=1.0)   # mid

for i, r in enumerate(table_rows):
    yy = y0 - 0.06 - dy*(i+1) + 0.04
    for j, txt in enumerate(r):
        ha = "left" if j <= 1 else "center"
        m = header[j] if j >= 2 else None
        bold = (m is not None and abs(C[r[0]][m] - colmax[m]) < 1e-9)
        ax.text(xs[j], yy, txt, fontsize=12, ha=ha, va="center",
                fontweight="bold" if bold else "normal",
                color="#0a4d8c" if bold else "black")
# bottom rule
ybot = y0 - 0.06 - dy*len(table_rows) + 0.01
ax.plot([0, 1.0], [ybot, ybot], color="black", lw=1.6)

ax.text(0.0, ybot-0.1,
        "OpenWikiTable (24,680 tables), BM25, n=1000 queries, seed=42.  "
        "Bold = best per column.  R@k = gold table within top-k; MRR = mean reciprocal rank.",
        fontsize=8.5, color="#444", ha="left", va="top")
ax.set_xlim(-0.02, 1.0); ax.set_ylim(ybot-0.2, 1.0)

fig.savefig(base / "docs/table_bm25_results.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("saved")
for r in table_rows:
    print(r)
