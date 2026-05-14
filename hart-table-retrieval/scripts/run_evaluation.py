#!/usr/bin/env python3
"""Evaluate retrieval results and produce summary CSV."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_result_filename(filename: str) -> dict:
    """Parse serializer_model_alpha.json filename."""
    name = filename.replace(".json", "")
    # Try to split by last underscore for alpha
    parts = name.rsplit("_", 1)
    try:
        alpha = float(parts[-1])
        rest = parts[0]
    except (ValueError, IndexError):
        alpha = 0.0
        rest = name

    # Split rest into serializer and model
    for ser in ["header_path", "plain_markdown", "json_kv"]:
        if rest.startswith(ser + "_"):
            model = rest[len(ser) + 1:]
            return {"serializer": ser, "model": model, "alpha": alpha}

    return {"serializer": rest, "model": "unknown", "alpha": alpha}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/evaluation_summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    result_files = sorted(results_dir.glob("*.json"))

    # Filter out non-retrieval result files
    result_files = [
        f for f in result_files
        if not f.name.startswith("indexing_")
        and not f.name.startswith("ablation_")
        and not f.name.startswith("token_")
    ]

    if not result_files:
        logger.error("No result files found in %s", results_dir)
        return

    all_rows = []

    for rf in result_files:
        info = parse_result_filename(rf.name)

        with open(rf, "r") as f:
            data = json.load(f)

        if not data:
            continue

        all_predictions = [d["predicted"] for d in data]
        all_relevants = [d["relevant"] for d in data]

        metrics = evaluate_batch(all_predictions, all_relevants, ks=[1, 5, 10])

        row = {
            "serializer": info["serializer"],
            "model": info["model"],
            "alpha": info["alpha"],
            **metrics,
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["serializer", "model", "alpha"]).reset_index(drop=True)

    df.to_csv(args.output, index=False)
    logger.info("Saved evaluation summary to %s", args.output)

    # Pretty print
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
