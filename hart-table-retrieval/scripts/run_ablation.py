#!/usr/bin/env python3
"""Run HART ablation experiments."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import chromadb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.loader import load_hitab, get_query_from_sample, get_table_id
from src.retrieval.embedder import EmbedderFactory
from src.retrieval.hart_scorer import HARTScorer
from src.retrieval.searcher import TableSearcher
from src.retrieval.indexer import _sanitize_collection_name, _model_short_name
from src.evaluation.metrics import evaluate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class HARTScorerNoDepth(HARTScorer):
    """Ablation: uniform weights instead of k/d depth weighting."""

    def compute_structural_alignment(self, query_embedding, path, depth):
        if not path or depth == 0:
            return 0.0
        d = depth
        total = 0.0
        for header in path:
            h_emb = self._get_header_embedding(header)
            cos_sim = self._cosine_similarity(query_embedding, h_emb)
            total += cos_sim * 1.0  # uniform weight
        return total / d


def run_variant(
    name, collection, embedder, qa_pairs, scorer, top_k_vectors=50, top_k_tables=10
):
    """Run a single ablation variant."""
    searcher = TableSearcher(collection, embedder, scorer)
    predictions = []
    relevants = []

    for qa in qa_pairs:
        results = searcher.search(qa["query"], top_k_vectors, top_k_tables)
        predicted = [r["table_id"] for r in results]
        predictions.append(predicted)
        relevants.append(qa["relevant"])

    metrics = evaluate_batch(predictions, relevants, ks=[1, 5, 10])
    metrics["variant"] = name
    return metrics


def main():
    parser = argparse.ArgumentParser(description="HART ablation experiments")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data-dir", default="data/hitab")
    parser.add_argument("--chroma-dir", default="./chroma_db")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--model", default=None, help="Specific model to use")
    args = parser.parse_args()

    config = load_config(args.config)

    # Pick embedding model
    if args.model:
        model_config = next(
            (m for m in config["embedding_models"] if m["name"] == args.model), None
        )
    else:
        # Use first available sentence-transformer
        model_config = next(
            (m for m in config["embedding_models"] if m["type"] == "sentence-transformer"),
            config["embedding_models"][0],
        )

    embedder = EmbedderFactory.create(model_config)
    if embedder is None:
        logger.error("Could not create embedder for %s", model_config["name"])
        return

    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)
    model_short = _model_short_name(model_config["name"])

    # Load QA pairs
    samples = load_hitab(data_dir=args.data_dir, max_samples=args.max_queries)
    qa_pairs = []
    for s in samples:
        query = get_query_from_sample(s)
        table_id = get_table_id(s)
        if query and table_id:
            qa_pairs.append({"query": query, "relevant": table_id})

    logger.info("QA pairs: %d, Model: %s", len(qa_pairs), model_config["name"])

    # Get collections
    hp_col_name = _sanitize_collection_name(f"header_path_{model_short}")
    pm_col_name = _sanitize_collection_name(f"plain_markdown_{model_short}")

    try:
        hp_collection = chroma_client.get_collection(hp_col_name)
    except Exception:
        logger.error("header_path collection not found: %s", hp_col_name)
        return

    try:
        pm_collection = chroma_client.get_collection(pm_col_name)
    except Exception:
        logger.error("plain_markdown collection not found: %s", pm_col_name)
        return

    # Find best alpha from existing results
    best_alpha = 0.5
    results_dir = Path("results")
    for alpha in config["hart"]["alpha_values"]:
        rf = results_dir / f"header_path_{model_short}_{alpha:.1f}.json"
        if rf.exists():
            # Could load and check, for now just use 0.5
            pass

    all_results = []

    # 1. HART-full
    scorer_full = HARTScorer(embedder, alpha=best_alpha)
    metrics = run_variant("HART-full", hp_collection, embedder, qa_pairs, scorer_full)
    all_results.append(metrics)
    logger.info("HART-full: %s", {k: f"{v:.4f}" for k, v in metrics.items() if k != "variant"})

    # 2. HART-no-align (alpha=1.0, content sim only)
    scorer_no_align = HARTScorer(embedder, alpha=1.0)
    metrics = run_variant("HART-no-align", hp_collection, embedder, qa_pairs, scorer_no_align)
    all_results.append(metrics)
    logger.info("HART-no-align: %s", {k: f"{v:.4f}" for k, v in metrics.items() if k != "variant"})

    # 3. HART-no-depth (uniform weights)
    scorer_no_depth = HARTScorerNoDepth(embedder, alpha=best_alpha)
    metrics = run_variant("HART-no-depth", hp_collection, embedder, qa_pairs, scorer_no_depth)
    all_results.append(metrics)
    logger.info("HART-no-depth: %s", {k: f"{v:.4f}" for k, v in metrics.items() if k != "variant"})

    # 4. HART-single-vec (PlainMarkdown serializer)
    metrics = run_variant("HART-single-vec", pm_collection, embedder, qa_pairs, None)
    all_results.append(metrics)
    logger.info("HART-single-vec: %s", {k: f"{v:.4f}" for k, v in metrics.items() if k != "variant"})

    # Save
    df = pd.DataFrame(all_results)
    cols = ["variant"] + [c for c in df.columns if c != "variant"]
    df = df[cols]
    df.to_csv(results_dir / "ablation_summary.csv", index=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== Ablation Results ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
