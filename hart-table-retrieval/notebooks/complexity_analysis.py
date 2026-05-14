#!/usr/bin/env python3
"""Complexity-based analysis of HART retrieval results."""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_hitab, get_table_from_sample, get_table_id
from src.data.header_tree import HeaderTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def classify_complexity(depth: int, span_ratio: float) -> str:
    if depth <= 2 and span_ratio == 0:
        return "Simple"
    elif depth <= 3 and span_ratio < 0.3:
        return "Medium"
    else:
        return "Complex"


def main():
    # 1. Load tables and classify complexity
    logger.info("Loading HiTab tables...")
    samples = load_hitab()

    table_complexity = {}
    for s in samples:
        tid = get_table_id(s)
        if tid in table_complexity:
            continue
        table = get_table_from_sample(s)
        tree = HeaderTree()
        tree.build_tree(table)
        depth = tree.get_depth()
        span_ratio = tree.get_spanning_cell_ratio()
        table_complexity[tid] = {
            "depth": depth,
            "span_ratio": span_ratio,
            "group": classify_complexity(depth, span_ratio),
        }

    # Count groups
    groups = {"Simple": 0, "Medium": 0, "Complex": 0}
    for info in table_complexity.values():
        groups[info["group"]] += 1
    logger.info("Group counts: %s", groups)

    # 2. Load evaluation summary
    eval_path = Path("results/evaluation_summary.csv")
    if not eval_path.exists():
        logger.error("evaluation_summary.csv not found. Run evaluation first.")
        return

    eval_df = pd.read_csv(eval_path)

    # 3. Load per-query results and map to complexity groups
    results_dir = Path("results")
    result_files = sorted(results_dir.glob("*.json"))
    result_files = [
        f for f in result_files
        if not f.name.startswith("indexing_")
        and not f.name.startswith("ablation_")
        and not f.name.startswith("token_")
    ]

    group_results = []

    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)

        for entry in data:
            tid = entry["relevant"]
            info = table_complexity.get(tid, {"group": "Unknown"})
            predicted = entry["predicted"]
            # Compute Recall@5 and nDCG@10 per query
            recall5 = 1.0 if tid in predicted[:5] else 0.0
            # nDCG@10
            dcg = 0.0
            for i, p in enumerate(predicted[:10]):
                if p == tid:
                    dcg = 1.0 / np.log2(i + 2)
                    break
            ndcg10 = dcg / (1.0 / np.log2(2))

            group_results.append({
                "file": rf.name,
                "group": info["group"],
                "Recall@5": recall5,
                "nDCG@10": ndcg10,
                "table_id": tid,
            })

    gdf = pd.DataFrame(group_results)

    # 4. Group-level stats
    print("\n=== Group-level Performance ===")
    for grp in ["Simple", "Medium", "Complex"]:
        subset = gdf[gdf["group"] == grp]
        if len(subset) == 0:
            continue
        print(f"\n{grp} (n={len(subset)}):")
        by_file = subset.groupby("file")[["Recall@5", "nDCG@10"]].mean()
        print(by_file.to_string())

    # 5. Delta: HART vs PlainMarkdown
    hart_files = [f for f in result_files if "header_path" in f.name]
    pm_files = [f for f in result_files if "plain_markdown" in f.name]

    if hart_files and pm_files:
        print("\n=== HART vs PlainMarkdown Delta by Complexity ===")
        for grp in ["Simple", "Medium", "Complex"]:
            hart_data = gdf[(gdf["group"] == grp) & gdf["file"].str.contains("header_path")]
            pm_data = gdf[(gdf["group"] == grp) & gdf["file"].str.contains("plain_markdown")]

            if len(hart_data) > 0 and len(pm_data) > 0:
                hart_r5 = hart_data["Recall@5"].mean()
                pm_r5 = pm_data["Recall@5"].mean()
                delta = hart_r5 - pm_r5
                print(f"  {grp}: HART={hart_r5:.4f}, PM={pm_r5:.4f}, delta={delta:+.4f}")

    # 6. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Grouped bar chart
    group_names = ["Simple", "Medium", "Complex"]
    methods = gdf["file"].unique()[:4]  # first 4

    x = np.arange(len(group_names))
    width = 0.2
    for i, method in enumerate(methods):
        vals = []
        for grp in group_names:
            subset = gdf[(gdf["file"] == method) & (gdf["group"] == grp)]
            vals.append(subset["Recall@5"].mean() if len(subset) > 0 else 0)
        label = method.replace(".json", "")[:30]
        axes[0].bar(x + i * width, vals, width, label=label)

    axes[0].set_xlabel("Complexity Group")
    axes[0].set_ylabel("Recall@5")
    axes[0].set_title("Recall@5 by Complexity Group")
    axes[0].set_xticks(x + width * len(methods) / 2)
    axes[0].set_xticklabels(group_names)
    axes[0].legend(fontsize=7)

    # Line chart: alpha vs Recall@5 by complexity
    alpha_files = sorted([f for f in result_files if "header_path" in f.name])
    if alpha_files:
        for grp in group_names:
            alphas = []
            recalls = []
            for af in alpha_files:
                try:
                    alpha_val = float(af.stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                subset = gdf[(gdf["file"] == af.name) & (gdf["group"] == grp)]
                if len(subset) > 0:
                    alphas.append(alpha_val)
                    recalls.append(subset["Recall@5"].mean())
            if alphas:
                axes[1].plot(alphas, recalls, marker="o", label=grp)

        axes[1].set_xlabel("Alpha")
        axes[1].set_ylabel("Recall@5")
        axes[1].set_title("Recall@5 vs Alpha by Complexity")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/complexity_analysis.png", dpi=150)
    logger.info("Saved plot to results/complexity_analysis.png")

    # 7. Statistical tests
    print("\n=== Statistical Tests ===")
    hart_subset = gdf[gdf["file"].str.contains("header_path")]
    pm_subset = gdf[gdf["file"].str.contains("plain_markdown")]

    if len(hart_subset) > 0 and len(pm_subset) > 0:
        # Wilcoxon: paired by table_id
        merged = hart_subset.merge(
            pm_subset, on="table_id", suffixes=("_hart", "_pm")
        )
        if len(merged) > 10:
            diff = merged["Recall@5_hart"] - merged["Recall@5_pm"]
            diff_nonzero = diff[diff != 0]
            if len(diff_nonzero) > 0:
                stat_w, p_w = stats.wilcoxon(diff_nonzero)
                print(f"  Wilcoxon (HART vs PM): stat={stat_w:.4f}, p={p_w:.6f}")

        # Kruskal-Wallis: delta across groups
        deltas_by_group = {}
        for grp in group_names:
            g_merged = merged[merged["group_hart"] == grp]
            if len(g_merged) > 0:
                deltas_by_group[grp] = (
                    g_merged["Recall@5_hart"] - g_merged["Recall@5_pm"]
                ).values

        if len(deltas_by_group) >= 2:
            groups_list = [v for v in deltas_by_group.values() if len(v) > 0]
            if len(groups_list) >= 2:
                stat_k, p_k = stats.kruskal(*groups_list)
                print(f"  Kruskal-Wallis (delta across groups): stat={stat_k:.4f}, p={p_k:.6f}")


if __name__ == "__main__":
    main()
