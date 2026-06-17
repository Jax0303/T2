#!/usr/bin/env python3
"""Download the HiTab benchmark into the layout ``rag_agent.data.loader`` expects.

The fresh-clone container does not ship the dataset (the old ``data/hitab``
symlink pointed outside the repo). This script fetches the official
microsoft/HiTab release and writes::

    data/hitab/data/
        train_samples.jsonl
        dev_samples.jsonl
        test_samples.jsonl
        tables/hmt/<table_id>.json
        tables/raw/<table_id>.json

Usage
-----
    python data/download_hitab.py                 # default: data/hitab
    python data/download_hitab.py --dest data/hitab --force

It is idempotent: existing files are skipped unless ``--force`` is given.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/microsoft/HiTab/main/data"
SAMPLE_FILES = ["train_samples.jsonl", "dev_samples.jsonl", "test_samples.jsonl"]
TABLES_ZIP = "tables.zip"
UA = {"User-Agent": "table-rag-pipeline/1.0"}


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_samples(dest_data: Path, force: bool) -> None:
    for name in SAMPLE_FILES:
        out = dest_data / name
        if out.exists() and not force:
            print(f"  skip   {out} (exists)")
            continue
        print(f"  fetch  {name} ...", flush=True)
        out.write_bytes(_fetch(f"{RAW_BASE}/{name}"))
        n = sum(1 for _ in out.open(encoding="utf-8"))
        print(f"  wrote  {out} ({n} samples)")


def download_tables(dest_data: Path, force: bool) -> None:
    marker = dest_data / "tables" / "hmt"
    if marker.exists() and any(marker.iterdir()) and not force:
        print(f"  skip   {dest_data / 'tables'} (already extracted)")
        return
    print(f"  fetch  {TABLES_ZIP} ...", flush=True)
    blob = _fetch(f"{RAW_BASE}/{TABLES_ZIP}")
    print(f"  unzip  {len(blob)/1e6:.1f} MB -> {dest_data}", flush=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest_data)  # archive root is 'tables/{hmt,raw}/...'
    n_hmt = len(list((dest_data / "tables" / "hmt").glob("*.json")))
    n_raw = len(list((dest_data / "tables" / "raw").glob("*.json")))
    print(f"  wrote  tables/hmt ({n_hmt}) + tables/raw ({n_raw})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data/hitab", help="destination root (default: data/hitab)")
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args(argv)

    dest_data = Path(args.dest) / "data"
    (dest_data / "tables").mkdir(parents=True, exist_ok=True)

    print(f"HiTab -> {dest_data}")
    download_samples(dest_data, args.force)
    download_tables(dest_data, args.force)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
