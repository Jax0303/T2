#!/usr/bin/env python3
"""Index HiTab tables into ChromaDB for all serializer x model combinations."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.loader import load_hitab, get_table_from_sample, get_table_id
from src.data.header_tree import HeaderTree
from src.serializers.plain_markdown import PlainMarkdownSerializer
from src.serializers.json_kv import JsonKeyValueSerializer
from src.serializers.header_path import HeaderPathSerializer
from src.retrieval.embedder import EmbedderFactory
from src.retrieval.indexer import TableIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SERIALIZER_MAP = {
    "plain_markdown": PlainMarkdownSerializer,
    "json_kv": JsonKeyValueSerializer,
    "header_path": HeaderPathSerializer,
}


def main():
    parser = argparse.ArgumentParser(description="Index HiTab tables")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data-dir", default="data/hitab")
    parser.add_argument("--chroma-dir", default="./chroma_db")
    parser.add_argument("--max-tables", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)

    # Load data
    logger.info("Loading HiTab data...")
    samples = load_hitab(data_dir=args.data_dir, max_samples=args.max_tables)
    logger.info("Loaded %d samples", len(samples))

    # Deduplicate tables
    tables = {}
    for sample in samples:
        tid = get_table_id(sample)
        if tid not in tables:
            tables[tid] = get_table_from_sample(sample)
            tables[tid]["uid"] = tid

    logger.info("Unique tables: %d", len(tables))

    stats = []

    for ser_name in config["serializers"]:
        serializer = SERIALIZER_MAP[ser_name]()

        # Pre-serialize all tables
        logger.info("Serializing with %s...", ser_name)
        t0 = time.time()

        all_docs = []
        for tid, table in tables.items():
            tree = HeaderTree()
            tree.build_tree(table)
            docs = serializer.serialize(table, tree)
            all_docs.extend(docs)

        ser_time = time.time() - t0
        logger.info(
            "  %s: %d docs from %d tables (%.1fs)",
            ser_name, len(all_docs), len(tables), ser_time,
        )

        for model_config in config["embedding_models"]:
            embedder = EmbedderFactory.create(model_config)
            if embedder is None:
                logger.warning("Skipping %s (embedder unavailable)", model_config["name"])
                continue

            logger.info("Indexing %s + %s...", ser_name, model_config["name"])
            t0 = time.time()

            try:
                indexer = TableIndexer(
                    chroma_client, embedder, ser_name, model_config["name"]
                )
                n_indexed = indexer.index_documents(all_docs)
            except Exception as e:
                logger.error(
                    "Indexing failed for %s + %s (%s: %s) — skipping",
                    ser_name, model_config["name"], type(e).__name__, str(e)[:200],
                )
                continue
            idx_time = time.time() - t0

            stat = indexer.get_stats()
            stat["serialization_time"] = ser_time
            stat["indexing_time"] = idx_time
            stat["avg_docs_per_table"] = len(all_docs) / max(len(tables), 1)
            stats.append(stat)

            logger.info(
                "  Done: %d vectors in %.1fs", n_indexed, idx_time
            )

    # Save stats
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "indexing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n=== Indexing Summary ===")
    for s in stats:
        logger.info(
            "  %s | %s: %d vectors, %.1f docs/table",
            s["serializer"], s["model"],
            s["total_vectors"], s["avg_docs_per_table"],
        )


if __name__ == "__main__":
    main()
