#!/usr/bin/env python3
"""Token length confound control analysis."""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_hitab, get_table_from_sample, get_table_id
from src.data.header_tree import HeaderTree
from src.serializers.plain_markdown import PlainMarkdownSerializer
from src.serializers.json_kv import JsonKeyValueSerializer
from src.serializers.header_path import HeaderPathSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    # Load tables
    logger.info("Loading HiTab tables...")
    samples = load_hitab()

    tables = {}
    for s in samples:
        tid = get_table_id(s)
        if tid not in tables:
            tables[tid] = get_table_from_sample(s)
            tables[tid]["uid"] = tid

    logger.info("Unique tables: %d", len(tables))

    serializers = {
        "plain_markdown": PlainMarkdownSerializer(),
        "json_kv": JsonKeyValueSerializer(),
        "header_path": HeaderPathSerializer(),
    }

    # 1. Token counts per serializer
    token_stats = {}
    table_tokens = {}  # {ser_name: {table_id: token_count_or_list}}

    for ser_name, serializer in serializers.items():
        all_tokens = []
        sub_doc_tokens = []  # for header_path
        table_tokens[ser_name] = {}

        for tid, table in tables.items():
            tree = HeaderTree()
            tree.build_tree(table)
            docs = serializer.serialize(table, tree)

            total = 0
            for text, meta in docs:
                n_tok = count_tokens(text, enc)
                total += n_tok
                if ser_name == "header_path":
                    sub_doc_tokens.append(n_tok)

            all_tokens.append(total)
            table_tokens[ser_name][tid] = total

        token_stats[ser_name] = {
            "mean": np.mean(all_tokens),
            "median": np.median(all_tokens),
            "min": np.min(all_tokens),
            "max": np.max(all_tokens),
            "std": np.std(all_tokens),
        }

        if ser_name == "header_path" and sub_doc_tokens:
            token_stats[ser_name]["sub_doc_mean"] = np.mean(sub_doc_tokens)
            token_stats[ser_name]["sub_doc_median"] = np.median(sub_doc_tokens)
            token_stats[ser_name]["sub_doc_min"] = np.min(sub_doc_tokens)
            token_stats[ser_name]["sub_doc_max"] = np.max(sub_doc_tokens)

    # Print stats
    print("\n=== Token Count Statistics ===")
    for ser_name, s in token_stats.items():
        print(f"\n{ser_name}:")
        for k, v in s.items():
            print(f"  {k}: {v:.1f}")

    # 2. Correlation: token count vs retrieval performance
    results_dir = Path("results")
    result_files = sorted(results_dir.glob("*.json"))
    result_files = [
        f for f in result_files
        if not f.name.startswith("indexing_")
        and not f.name.startswith("ablation_")
        and not f.name.startswith("token_")
    ]

    scatter_data = []
    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)

        # Determine serializer from filename
        ser_name = None
        for s in ["header_path", "plain_markdown", "json_kv"]:
            if s in rf.name:
                ser_name = s
                break
        if ser_name is None:
            continue

        for entry in data:
            tid = entry["relevant"]
            predicted = entry["predicted"]

            # reciprocal rank
            rr = 0.0
            for i, p in enumerate(predicted):
                if p == tid:
                    rr = 1.0 / (i + 1)
                    break

            tok_count = table_tokens.get(ser_name, {}).get(tid, 0)
            if tok_count > 0:
                scatter_data.append({
                    "serializer": ser_name,
                    "file": rf.name,
                    "table_id": tid,
                    "token_count": tok_count,
                    "reciprocal_rank": rr,
                })

    if scatter_data:
        sdf = pd.DataFrame(scatter_data)

        print("\n=== Token Count vs Retrieval Performance (Spearman) ===")
        for ser_name in sdf["serializer"].unique():
            subset = sdf[sdf["serializer"] == ser_name]
            if len(subset) > 5:
                corr, p_val = stats.spearmanr(
                    subset["token_count"], subset["reciprocal_rank"]
                )
                print(f"  {ser_name}: rho={corr:.4f}, p={p_val:.6f} (n={len(subset)})")

        # Scatter plot
        fig, axes = plt.subplots(1, len(serializers), figsize=(5 * len(serializers), 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, ser_name in zip(axes, serializers.keys()):
            subset = sdf[sdf["serializer"] == ser_name]
            if len(subset) > 0:
                ax.scatter(
                    subset["token_count"], subset["reciprocal_rank"],
                    alpha=0.3, s=10
                )
                ax.set_xlabel("Token Count")
                ax.set_ylabel("Reciprocal Rank")
                ax.set_title(ser_name)

        plt.tight_layout()
        plt.savefig("results/token_length_scatter.png", dpi=150)
        logger.info("Saved scatter plot to results/token_length_scatter.png")

    # 3. Save token analysis CSV
    rows = []
    for ser_name, s in token_stats.items():
        row = {"serializer": ser_name, **s}
        rows.append(row)
    pd.DataFrame(rows).to_csv("results/token_length_analysis.csv", index=False)
    logger.info("Saved token length analysis to results/token_length_analysis.csv")

    # 4. Truncated PlainMarkdown baseline info
    if "header_path" in token_stats:
        avg_sub_len = token_stats["header_path"].get("sub_doc_mean", 50)
        print(f"\n=== PlainMarkdown-Truncated ===")
        print(f"  Target truncation length: {avg_sub_len:.0f} tokens")
        print(
            "  To run this baseline, re-index PlainMarkdown with text truncated to "
            f"{avg_sub_len:.0f} tokens per table, then re-evaluate."
        )


if __name__ == "__main__":
    main()
