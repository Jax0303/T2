#!/usr/bin/env python3
"""Download OpenWikiTable and normalize it for the preprocessing experiment.

Produces under --out-dir (default rag-agent/data/openwikitable):

  corpus.jsonl          one record per table (24,680):
                        {table_id, page_title, section_title, caption,
                         header, rows, name, dataset}
  queries_test.jsonl    {question_id, question, gold_table_id, dataset}
  queries_valid.jsonl   same for the validation split

Source: https://github.com/sean0042/Open_WikiTable (Kweon et al., 2023,
Findings of ACL — CC BY-SA). The repo ships data/data.tar.gz; we clone
shallow, extract, and convert the pandas-orient JSON to jsonl.

The retrieval task: given `question`, rank the gold table (original_table_id)
within the full 24,680-table corpus.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_URL = "https://github.com/sean0042/Open_WikiTable.git"


def fetch(work_dir: Path) -> Path:
    data_dir = work_dir / "Open_WikiTable" / "data"
    if not (data_dir / "data.tar.gz").exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL],
            cwd=work_dir, check=True,
        )
    if not (data_dir / "tables.json").exists():
        with tarfile.open(data_dir / "data.tar.gz") as tf:
            tf.extractall(data_dir)
    return data_dir


def convert(data_dir: Path, out_dir: Path) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    tables = pd.read_json(data_dir / "tables.json")
    with open(out_dir / "corpus.jsonl", "w") as f:
        for _, r in tables.iterrows():
            f.write(json.dumps({
                "table_id": r["original_table_id"],
                "page_title": r["page_title"],
                "section_title": r["section_title"],
                "caption": r["caption"],
                "header": r["header"],
                "rows": r["rows"],
                "name": r["name"],
                "dataset": r["dataset"],
            }, ensure_ascii=False) + "\n")
    print(f"corpus.jsonl: {len(tables)} tables")

    corpus_ids = set(tables["original_table_id"])
    for split in ["valid", "test"]:
        q = pd.read_json(data_dir / f"{split}.json")
        n_orphan = 0
        with open(out_dir / f"queries_{split}.jsonl", "w") as f:
            for _, r in q.iterrows():
                if r["original_table_id"] not in corpus_ids:
                    n_orphan += 1
                    continue
                f.write(json.dumps({
                    "question_id": r["question_id"],
                    "question": r["question"],
                    "gold_table_id": r["original_table_id"],
                    "dataset": r["dataset"],
                }, ensure_ascii=False) + "\n")
        print(f"queries_{split}.jsonl: {len(q) - n_orphan} queries "
              f"(orphans dropped: {n_orphan})")
        if n_orphan:
            print(f"  WARNING: {n_orphan} {split} questions reference "
                  f"tables missing from the corpus", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="/tmp/owt_src",
                   help="where to clone/extract the upstream repo")
    p.add_argument("--out-dir", default="rag-agent/data/openwikitable")
    args = p.parse_args()

    data_dir = fetch(Path(args.work_dir))
    convert(data_dir, Path(args.out_dir))


if __name__ == "__main__":
    main()
