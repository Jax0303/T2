#!/usr/bin/env python3
"""Fetch the benchmarks used in the thesis experiments.

* **HiTab** (core target, hierarchical tables) — direct download, no extra deps.
* **FinQA** (numeric reasoning) — via HuggingFace ``datasets`` (optional).
* **WikiSQL** (flat-table control) — via HuggingFace ``datasets`` (optional).

HiTab always runs. FinQA / WikiSQL are pulled only when ``datasets`` is
installed and the corresponding flag is passed, so the pipeline can be set up
on the core benchmark without the heavier HF dependency.

Usage
-----
    python data/download_benchmarks.py                 # HiTab only
    python data/download_benchmarks.py --finqa --wikisql
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import download_hitab  # noqa: E402  (same directory)


def download_finqa(dest: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  FinQA: `datasets` not installed — `pip install datasets` to enable. Skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("  FinQA: loading dnagpt/finqa via datasets ...", flush=True)
    ds = load_dataset("dreamerdeo/finqa")  # train/dev/test with `table`, `qa`
    for split in ds:
        out = dest / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for rec in ds[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {out} ({len(ds[split])} records)")


def download_wikisql(dest: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  WikiSQL: `datasets` not installed — `pip install datasets` to enable. Skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("  WikiSQL: loading Salesforce/wikisql via datasets ...", flush=True)
    ds = load_dataset("Salesforce/wikisql")
    for split in ds:
        out = dest / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for rec in ds[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {out} ({len(ds[split])} records)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data", help="benchmark root dir (default: data)")
    ap.add_argument("--finqa", action="store_true", help="also download FinQA (needs `datasets`)")
    ap.add_argument("--wikisql", action="store_true", help="also download WikiSQL (needs `datasets`)")
    ap.add_argument("--force", action="store_true", help="re-download HiTab even if present")
    args = ap.parse_args(argv)

    root = Path(args.root)

    print("== HiTab ==")
    download_hitab.main(["--dest", str(root / "hitab")] + (["--force"] if args.force else []))

    if args.finqa:
        print("== FinQA ==")
        download_finqa(root / "finqa")
    if args.wikisql:
        print("== WikiSQL ==")
        download_wikisql(root / "wikisql")

    print("all done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
