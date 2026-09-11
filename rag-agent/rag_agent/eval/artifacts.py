# SPDX-License-Identifier: MIT
"""Integrity checks for the evidence that is scored and delivered to a reader."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CONTEXT_VERSION = 2


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(path) -> dict:
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty record")
            row = json.loads(line)
            qid = row["query_id"]
            if qid in records:
                raise ValueError(f"{path}:{line_number}: duplicate query_id {qid}")
            records[qid] = row
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def require_same_ids(reference, other, label="records"):
    missing, extra = set(reference) - set(other), set(other) - set(reference)
    if missing or extra:
        raise ValueError(f"{label}: query IDs differ (missing={len(missing)}, extra={len(extra)})")


@dataclass
class Selection:
    cells: set
    units: list
    dump: bool

    @property
    def context(self):
        return [u["text"] for u in self.units] if self.dump else []

    def __iter__(self):
        # Preserve the three-value API used by existing analysis scripts.
        yield self.cells
        yield len(self.cells)
        yield self.context


def evidence_fields(selection: Selection, gold: set) -> dict:
    unit_cells = set().union(*(set(map(tuple, u["cells"])) for u in selection.units))
    if unit_cells != selection.cells:
        raise ValueError("selected cells differ from delivered units")
    return {"context_version": CONTEXT_VERSION,
            "context": selection.context,
            "context_units": selection.units,
            "context_cells": sorted(selection.cells),
            "context_sha256": digest(selection.context),
            "evidence_sha256": digest(selection.units),
            "gold_cells": sorted(gold)}


def validate_retrieval(row):
    if "correct" not in row:
        if not row.get("excluded"):
            raise ValueError("unscored record must state an exclusion reason")
        return
    if row.get("context_version") != CONTEXT_VERSION:
        raise ValueError(f"{row['query_id']}: legacy evidence needs regeneration (context_version=2)")
    ctx, units = row["context"], row["context_units"]
    coordinates = [*row["context_cells"], *row["gold_cells"],
                   *(c for u in units for c in u["cells"])]
    for c in coordinates:
        if (not isinstance(c, (list, tuple)) or len(c) != 3 or not isinstance(c[0], str)
                or not c[0] or any(type(v) is not int or v < 0 for v in c[1:])):
            raise ValueError("invalid evidence coordinate")
    if any(c[0] != row["table_id"] for c in row["gold_cells"]):
        raise ValueError("gold coordinates belong to a different table")
    if ctx != [u["text"] for u in units]:
        raise ValueError("reader context differs from selected unit texts")
    if row["context_sha256"] != digest(ctx) or row["evidence_sha256"] != digest(units):
        raise ValueError("evidence hash mismatch")
    cells = set().union(*(set(map(tuple, u["cells"])) for u in units))
    listed = list(map(tuple, row["context_cells"]))
    gold_list = list(map(tuple, row["gold_cells"]))
    gold = set(gold_list)
    if len(listed) != len(set(listed)) or set(listed) != cells:
        raise ValueError("context cell coordinates are duplicated or incomplete")
    if row["cells_in_context"] != len(cells):
        raise ValueError("reported cell count differs from delivered evidence")
    if not gold or len(gold_list) != len(gold) or len(gold) != row["m"]:
        raise ValueError("gold coordinates are empty, duplicated, or truncated")
    if row["mode"] not in {"all", "any"}:
        raise ValueError("unknown retrieval scoring mode")
    hit = gold <= cells if row["mode"] == "all" else bool(gold & cells)
    if row["correct"] != int(hit):
        raise ValueError("retrieval score differs from delivered evidence")
    if row["gold_table_in_context"] != int(any(c[0] == row["table_id"] for c in cells)):
        raise ValueError("table hit differs from delivered evidence")


def provenance(root: Path) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    packages = {}
    for package in ("numpy", "scipy", "transformers", "sentence-transformers",
                    "langchain-text-splitters", "torch", "bitsandbytes"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    # A dirty checkout needs an exact code fingerprint in addition to HEAD.
    sources = {str(p.relative_to(root)).replace("\\", "/"): file_digest(p)
               for directory in ("rag_agent", "scripts", "analysis")
               for p in sorted((root / directory).rglob("*.py"))}
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "source_sha256": digest(sources), "python": platform.python_version(),
            "packages": packages}


def write_pair(records_path, rows, summary, *, summary_path=None):
    path = Path(records_path)
    if path.suffix != ".jsonl":
        raise ValueError("records output must end in .jsonl")
    meta = Path(summary_path) if summary_path is not None else path.with_suffix(".json")
    if meta == path:
        raise ValueError("records and summary paths must differ")
    if path.exists() or meta.exists():
        raise FileExistsError(f"output already exists: {path} or {meta}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {**summary, "records_sha256": file_digest(path)}
    with meta.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
