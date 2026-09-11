#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compose a diagnostic oracle from explicit, compatible v2 answer legs.

This selects existing predictions. It is not a fresh generation run or a
mathematical upper bound. Legacy gold files are deliberately unsupported.
"""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.validated_tables import read_leg
from rag_agent.eval.artifacts import file_digest, require_same_ids, write_pair
from rag_agent.eval.metrics import hitab_exact_match_text
from scripts.answer_accuracy import summarize


def compose(gold_path, retrieved_path, out):
    gold_path, retrieved_path = Path(gold_path), Path(retrieved_path)
    gold, gm = read_leg(gold_path, verified=True)
    retrieved, rm = read_leg(retrieved_path, verified=True)
    if gm.get("condition") != "gold" or rm.get("condition") != "retrieved":
        raise ValueError("expected gold and retrieved context conditions")
    require_same_ids(gold, retrieved, "oracle legs")
    for key in ("retrieval_records_sha256", "query_ids_sha256", "reader_details",
                "prompt_sha256", "seed", "max_new_tokens", "scorer",
                "excluded_unit_defect", "context_limit"):
        if key not in gm or key not in rm or gm[key] != rm[key]:
            raise ValueError(f"oracle legs disagree on required field {key}")
    rows = []
    for q, r in retrieved.items():
        g = gold[q]
        for key in ("retrieval_correct", "answer", "question", "mode", "aggregation", "source_context_sha256"):
            if key not in g or key not in r or g[key] != r[key]:
                raise ValueError(f"{q}: oracle legs disagree on {key}")
        for row in (g, r):
            if row["answer_correct"] != int(hitab_exact_match_text(row["pred"], row["answer"])):
                raise ValueError(f"{q}: stale answer score")
        rows.append(dict(g if r["retrieval_correct"] else r))
    summary = {**rm, "condition": "oracle", "composed": True,
               "composed_from": {"gold": str(gold_path), "retrieved": str(retrieved_path)},
               "composed_sha256": {"gold": file_digest(gold_path),
                                   "retrieved": file_digest(retrieved_path)},
               **summarize(rows, rm["context_limit"])}
    write_pair(out, rows, summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--retrieved", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    compose(args.gold, args.retrieved, args.out)


if __name__ == "__main__":
    main()
