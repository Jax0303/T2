#!/usr/bin/env python3
"""End-to-end small-scale test of the entire HART pipeline.

Pipeline:
  Data loading -> header tree -> serialization -> embedding -> indexing -> retrieval -> evaluation
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    load_hitab,
    get_table_from_sample,
    get_table_id,
    get_query_from_sample,
)
from src.data.header_tree import HeaderTree
from src.serializers.header_path import HeaderPathSerializer
from src.serializers.plain_markdown import PlainMarkdownSerializer
from src.retrieval.embedder import EmbedderFactory
from src.retrieval.hart_scorer import HARTScorer
from src.retrieval.indexer import TableIndexer
from src.retrieval.searcher import TableSearcher
from src.evaluation.metrics import evaluate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def time_step(name, fn):
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    logger.info("[%s] %.2fs", name, elapsed)
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/d/hart_data/hitab/HiTab")
    parser.add_argument("--chroma-dir", default="/mnt/d/hart_data/chroma_db_e2e")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model. Default is a small fast model.",
    )
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    # Set HF cache to D drive
    os.environ.setdefault("HF_HOME", "/mnt/d/hart_data/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/d/hart_data/hf_cache")

    # 1. Load data
    logger.info("Loading %d samples...", args.n_samples)
    samples, t_load = time_step(
        "load_data",
        lambda: load_hitab(args.data_dir, args.split, args.n_samples),
    )

    # Deduplicate tables
    tables = {}
    qa_pairs = []
    for s in samples:
        tid = get_table_id(s)
        if tid not in tables:
            t = get_table_from_sample(s)
            if t:
                tables[tid] = t
        q = get_query_from_sample(s)
        if q and tid in tables:
            qa_pairs.append({"query": q, "relevant": tid})

    logger.info(
        "Unique tables: %d, QA pairs: %d", len(tables), len(qa_pairs)
    )

    # 2. Header tree parsing + 3. Serialization
    serializer = HeaderPathSerializer()

    def _serialize_all():
        all_docs = []
        for tid, table in tables.items():
            tree = HeaderTree()
            tree.build_tree(table)
            docs = serializer.serialize(table, tree)
            all_docs.extend(docs)
        return all_docs

    all_docs, t_ser = time_step("serialize", _serialize_all)
    logger.info("Serialized to %d sub-documents", len(all_docs))

    # 4. Embedding
    if args.model in ("onnx-fallback", "onnx-default", "default"):
        embed_config = {"type": "onnx-default", "name": "chromadb-default"}
    else:
        embed_config = {"type": "sentence-transformer", "name": args.model}
    embedder, t_embed_init = time_step(
        "init_embedder",
        lambda: EmbedderFactory.create(embed_config),
    )
    if embedder is None:
        logger.error("Failed to create embedder")
        return

    # 5. Indexing
    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)
    # Clean any prior collection for the test
    try:
        chroma_client.delete_collection("e2e_test")
    except Exception:
        pass

    def _index():
        indexer = TableIndexer(
            chroma_client, embedder, "header_path", args.model
        )
        # Force a clean collection name
        idx = TableIndexer.__new__(TableIndexer)
        idx.client = chroma_client
        idx.embedder = embedder
        idx.serializer_name = "header_path"
        idx.model_name = args.model
        idx.collection = chroma_client.get_or_create_collection(
            name="e2e_test", metadata={"hnsw:space": "cosine"}
        )
        idx.index_documents(all_docs)
        return idx

    indexer, t_index = time_step("indexing", _index)

    # 6. Retrieval
    scorer = HARTScorer(embedder, alpha=args.alpha)
    searcher = TableSearcher(indexer.collection, embedder, scorer)

    def _retrieve():
        preds = []
        relevants = []
        for qa in qa_pairs:
            results = searcher.search(qa["query"], top_k_vectors=50, top_k_tables=10)
            preds.append([r["table_id"] for r in results])
            relevants.append(qa["relevant"])
        return preds, relevants

    (preds, relevants), t_retrieve = time_step("retrieval", _retrieve)

    # 7. Evaluation
    metrics, t_eval = time_step(
        "evaluation",
        lambda: evaluate_batch(preds, relevants, ks=[1, 5, 10]),
    )

    # Summary
    total = t_load + t_ser + t_embed_init + t_index + t_retrieve + t_eval
    print("\n=== Timing ===")
    print(f"  load_data:     {t_load:.2f}s")
    print(f"  serialize:     {t_ser:.2f}s")
    print(f"  init_embedder: {t_embed_init:.2f}s")
    print(f"  indexing:      {t_index:.2f}s")
    print(f"  retrieval:     {t_retrieve:.2f}s")
    print(f"  evaluation:    {t_eval:.2f}s")
    print(f"  TOTAL:         {total:.2f}s")

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nE2E test completed successfully.")


if __name__ == "__main__":
    main()
