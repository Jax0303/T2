# SPDX-License-Identifier: MIT
"""Load HiTab JSON tables into the internal Table representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.io.table_schema import Cell, HeaderNode, Table


def _parse_header_tree(node: dict[str, Any]) -> HeaderNode:
    """Recursively convert a HiTab TreeNode dict into a HeaderNode."""
    children = [_parse_header_tree(c) for c in node.get("children", [])]
    start = int(node.get("start_idx", -1))
    end = int(node.get("end_idx", -1))
    name = str(node.get("name", node.get("value", "")))
    return HeaderNode(name=name, span_start=start, span_end=end, children=children)


def _build_span_map(
    merged_regions: list[dict[str, int]],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Map (row, col) of a merge origin to (row_span, col_span)."""
    span_map: dict[tuple[int, int], tuple[int, int]] = {}
    for region in merged_regions:
        r1 = region["first_row"]
        c1 = region["first_column"]
        r2 = region["last_row"]
        c2 = region["last_column"]
        row_span = r2 - r1 + 1
        col_span = c2 - c1 + 1
        span_map[(r1, c1)] = (row_span, col_span)
    return span_map


def _cells_covered_by_merges(
    merged_regions: list[dict[str, int]],
) -> set[tuple[int, int]]:
    """Return the set of (row, col) positions that are non-origin parts of a merge."""
    covered: set[tuple[int, int]] = set()
    for region in merged_regions:
        r1 = region["first_row"]
        c1 = region["first_column"]
        r2 = region["last_row"]
        c2 = region["last_column"]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) != (r1, c1):
                    covered.add((r, c))
    return covered


def load_table(path: str | Path) -> Table:
    """Load a single HiTab table JSON file into a Table object.

    Args:
        path: Path to a HiTab table JSON file.

    Returns:
        Parsed Table with cells, header trees, and merged-cell info.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    texts: list[list[str]] = raw["texts"]
    n_top = int(raw.get("top_header_rows_num", 0))
    n_left = int(raw.get("left_header_columns_num", 0))
    merged_regions: list[dict[str, int]] = raw.get("merged_regions", [])

    span_map = _build_span_map(merged_regions)
    covered = _cells_covered_by_merges(merged_regions)

    merged_tuples: list[tuple[int, int, int, int]] = [
        (r["first_row"], r["first_column"], r["last_row"], r["last_column"])
        for r in merged_regions
    ]

    cells: list[list[Cell]] = []
    for r_idx, row in enumerate(texts):
        cell_row: list[Cell] = []
        for c_idx, val in enumerate(row):
            if (r_idx, c_idx) in covered:
                cell_row.append(Cell(value="", row_span=0, col_span=0, is_header=False))
                continue
            rs, cs = span_map.get((r_idx, c_idx), (1, 1))
            is_hdr = r_idx < n_top or c_idx < n_left
            cell_row.append(Cell(value=str(val), row_span=rs, col_span=cs, is_header=is_hdr))
        cells.append(cell_row)

    top_tree = _parse_header_tree(raw["top_root"]) if "top_root" in raw else HeaderNode(
        name="<ROOT>", span_start=-1, span_end=-1
    )
    left_tree = _parse_header_tree(raw["left_root"]) if "left_root" in raw else HeaderNode(
        name="<ROOT>", span_start=-1, span_end=-1
    )

    metadata: dict[str, Any] = {
        "table_id": raw.get("title", path.stem),
        "source_path": str(path),
        "top_header_rows_num": n_top,
        "left_header_columns_num": n_left,
    }

    return Table(
        cells=cells,
        top_header_tree=top_tree,
        left_header_tree=left_tree,
        merged_cells=merged_tuples,
        metadata=metadata,
    )


def load_tables(directory: str | Path, table_ids: list[str] | None = None) -> list[Table]:
    """Load multiple HiTab tables from a directory.

    Args:
        directory: Directory containing HiTab JSON files.
        table_ids: Optional list of table IDs (filenames without extension).
                   If None, loads all JSON files in the directory.

    Returns:
        List of parsed Table objects.
    """
    directory = Path(directory)
    if table_ids is not None:
        paths = [directory / f"{tid}.json" for tid in table_ids]
    else:
        paths = sorted(directory.glob("*.json"))

    return [load_table(p) for p in paths if p.exists()]
