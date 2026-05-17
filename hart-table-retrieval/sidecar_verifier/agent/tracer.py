"""Trace a free-form answer back to (row, col) cells in the structured table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..store.table_store import TableRecord
from .verifier import _parse_numbers  # noqa: re-export helper


@dataclass
class TraceResult:
    answer: str
    grounded_cells: List[Tuple[int, int, float]]   # row, col, value
    ungrounded_spans: List[float]
    grounded_fraction: float


def trace(answer: str, rec: TableRecord) -> TraceResult:
    numbers = _parse_numbers(answer)
    grounded: List[Tuple[int, int, float]] = []
    ungrounded: List[float] = []
    for n in numbers:
        cells = rec.find_value(n)
        if cells:
            r, c = cells[0]
            grounded.append((r, c, n))
        else:
            ungrounded.append(n)
    total = len(numbers)
    frac = (len(grounded) / total) if total else 1.0
    return TraceResult(
        answer=answer,
        grounded_cells=grounded,
        ungrounded_spans=ungrounded,
        grounded_fraction=frac,
    )
