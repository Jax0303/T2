# SPDX-License-Identifier: MIT
"""JSON-tree serializer — lossless (preserves header hierarchy + merges)."""

from __future__ import annotations

import json
from typing import Any

from src.io.table_schema import Cell, HeaderNode, Table
from src.serializers.base import SerializerBase


def _header_to_dict(node: HeaderNode) -> dict[str, Any]:
    """Recursively convert a HeaderNode to a JSON-serialisable dict."""
    return {
        "name": node.name,
        "span_start": node.span_start,
        "span_end": node.span_end,
        "children": [_header_to_dict(c) for c in node.children],
    }


def _dict_to_header(d: dict[str, Any]) -> HeaderNode:
    """Recursively reconstruct a HeaderNode from a dict."""
    return HeaderNode(
        name=d["name"],
        span_start=d["span_start"],
        span_end=d["span_end"],
        children=[_dict_to_header(c) for c in d.get("children", [])],
    )


class JsonTreeSerializer(SerializerBase):
    """Lossless JSON serializer that preserves header trees and merges."""

    # ------------------------------------------------------------------
    # serialize
    # ------------------------------------------------------------------
    def serialize(self, table: Table) -> str:
        """Serialize *table* to a JSON string."""
        cells_data: list[list[dict[str, Any]]] = []
        for row in table.cells:
            cells_data.append([
                {
                    "value": c.value,
                    "row_span": c.row_span,
                    "col_span": c.col_span,
                    "is_header": c.is_header,
                }
                for c in row
            ])
        payload: dict[str, Any] = {
            "cells": cells_data,
            "top_header_tree": _header_to_dict(table.top_header_tree),
            "left_header_tree": _header_to_dict(table.left_header_tree),
            "merged_cells": [list(m) for m in table.merged_cells],
            "metadata": table.metadata,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------
    def parse(self, text: str) -> Table:
        """Reconstruct a full Table (with trees and merges) from JSON."""
        data: dict[str, Any] = json.loads(text)

        cells: list[list[Cell]] = []
        for row in data["cells"]:
            cells.append([
                Cell(
                    value=c["value"],
                    row_span=c["row_span"],
                    col_span=c["col_span"],
                    is_header=c["is_header"],
                )
                for c in row
            ])

        return Table(
            cells=cells,
            top_header_tree=_dict_to_header(data["top_header_tree"]),
            left_header_tree=_dict_to_header(data["left_header_tree"]),
            merged_cells=[tuple(m) for m in data["merged_cells"]],
            metadata=data.get("metadata", {}),
        )
