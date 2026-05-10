# SPDX-License-Identifier: MIT
"""Internal Table data model for structural audit experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Cell:
    """Single table cell with span and header metadata."""

    value: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass
class HeaderNode:
    """Recursive tree node representing a hierarchical header."""

    name: str
    span_start: int
    span_end: int
    children: list[HeaderNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaves(self) -> list[HeaderNode]:
        """Return all leaf nodes in pre-order."""
        if self.is_leaf:
            return [self]
        result: list[HeaderNode] = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def ancestor_chain(self, target_idx: int) -> list[str]:
        """Return root-to-leaf name chain for a leaf at *target_idx*.

        Virtual root nodes (span_start == -1) always delegate to children.
        """
        is_virtual = self.span_start == -1 and self.span_end == -1
        in_range = is_virtual or (self.span_start <= target_idx <= self.span_end)
        if in_range:
            for child in self.children:
                sub = child.ancestor_chain(target_idx)
                if sub:
                    return [self.name] + sub
            if self.is_leaf and not is_virtual:
                return [self.name]
        return []


def _virtual_root() -> HeaderNode:
    return HeaderNode(name="<ROOT>", span_start=-1, span_end=-1)


@dataclass
class Table:
    """Unified table representation used across serializers and metrics."""

    cells: list[list[Cell]]
    top_header_tree: HeaderNode = field(default_factory=_virtual_root)
    left_header_tree: HeaderNode = field(default_factory=_virtual_root)
    merged_cells: list[tuple[int, int, int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.cells)

    @property
    def n_cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0
