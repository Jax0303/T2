#!/usr/bin/env python3
"""Run retrieval experiments for all serializer x model x alpha combinations."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.loader import load_hitab, get_query_from_sample, get_table_id
from src.retrieval.embedder import EmbedderFactory
from src.retrieval.hart_scorer import HARTScorer
from src.retrieval.searcher import TableSearcher
from src.retrieval.indexer import _sanitize_collection_name, _model_short_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run retrieval experiments")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data-dir", default="data/hitab")
    parser.add_argument("--chroma-dir", default="./chroma_db")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--top-k-vectors", type=int, default=50)
    parser.add_argument("--top-k-tables", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)

    # Load QA pairs
    logger.info("Loading QA pairs...")
    samples = load_hitab(data_dir=args.data_dir, max_samples=args.max_queries)

    qa_pairs = []
    for s in samples:
        query = get_query_from_sample(s)
        table_id = get_table_id(s)
        if query and table_id:
            qa_pairs.append({"query": query, "relevant": table_id})

    logger.info("QA pairs: %d", len(qa_pairs))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    alpha_values = config["hart"]["alpha_values"]

    for ser_name in config["serializers"]:
        for model_config in config["embedding_models"]:
            embedder = EmbedderFactory.create(model_config)
            if embedder is None:
                continue

            col_name = _sanitize_collection_name(
                f"{ser_name}_{_model_short_name(model_config['name'])}"
            )

            try:
                collection = chroma_client.get_collection(col_name)
            except Exception:
                logger.warning("Collection %s not found, skipping", col_name)
                continue

            for alpha in alpha_values:
                # Only use HART scorer for header_path serializer
                if ser_name == "header_path":
                    scorer = HARTScorer(embedder, alpha=alpha)
                else:
                    scorer = None

                searcher = TableSearcher(collection, embedder, scorer)

                logger.info(
                    "Retrieving: %s + %s + alpha=%.1f",
                    ser_name, model_config["name"], alpha,
                )

                t0 = time.time()
                run_results = []

                for qa in qa_pairs:
                    search_results = searcher.search(
                        qa["query"],
                        top_k_vectors=args.top_k_vectors,
                        top_k_tables=args.top_k_tables,
                    )
                    predicted = [r["table_id"] for r in search_results]
                    run_results.append({
                        "query": qa["query"],
                        "predicted": predicted,
                        "relevant": qa["relevant"],
                    })

                elapsed = time.time() - t0
                logger.info("  Done in %.1fs", elapsed)

                model_short = _model_short_name(model_config["name"])
                out_file = results_dir / f"{ser_name}_{model_short}_{alpha:.1f}.json"
                with open(out_file, "w") as f:
                    json.dump(run_results, f, indent=2)

                # For non-header_path serializers, alpha doesn't matter — run once
                if ser_name != "header_path":
                    break


if __name__ == "__main__":
    main()
