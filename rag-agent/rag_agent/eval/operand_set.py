# SPDX-License-Identifier: MIT
"""Operand-set completeness metrics for hierarchical-table aggregation retrieval.

The primary metric is **Operand-Set Completeness (OSC)**: a query scores 1 iff
*every* gold operand cell is present in the retrieved set (all-or-nothing subset
containment), and 0 otherwise. OSC is the necessary condition for a correct
aggregation answer and is strictly harder than averaged per-cell recall — a
query that misses a single operand still yields a wrong sum/avg/diff.

Cells are identified by ``(row, col)`` coordinates in *data space* (the same
space ``rag_agent.bench.hitab.resolve_gold_operands`` resolves operands into).
Gold/retrieved collections may be either ``(row, col)`` tuples or objects
exposing ``.row``/``.col`` (e.g. :class:`~rag_agent.bench.schema.GoldOperand`).

Stratification variables (per the research spec, §4):
  * ``scope_size`` m = number of distinct gold operand cells (aggregation scope).
  * ``header_depth`` d = max header-path length over the table's top/left trees.
  * ``aggregation`` = HiTab aggregation type (sum/avg/diff/count/...).

Note on naming: the retrieval *budget* (top-k) is ``k`` elsewhere in the repo;
here the aggregation *scope size* is ``m`` to avoid the collision flagged in the
spec review.
"""
from __future__ import annotations

from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

Cell = Tuple[int, int]


def _as_cellset(cells: Iterable) -> set:
    """Normalize a collection of cells to a set of ``(row, col)`` int tuples."""
    out: set = set()
    for c in cells:
        if isinstance(c, (tuple, list)):
            out.add((int(c[0]), int(c[1])))
        else:  # GoldOperand-like
            out.add((int(c.row), int(c.col)))
    return out


def operand_set_completeness(gold: Iterable, retrieved: Iterable) -> int:
    """1 iff every gold operand cell is in ``retrieved`` (all-or-nothing).

    Empty gold is treated as vacuously complete (returns 1); callers measuring
    OSC over a dataset should filter empty-gold queries upstream (see W1).
    """
    g = _as_cellset(gold)
    if not g:
        return 1
    return int(g.issubset(_as_cellset(retrieved)))


def per_cell_recall(gold: Iterable, retrieved: Iterable) -> float:
    """Fraction of gold operand cells present in ``retrieved`` (contrast metric)."""
    g = _as_cellset(gold)
    if not g:
        return 1.0
    return len(g & _as_cellset(retrieved)) / len(g)


def scope_size(gold: Iterable) -> int:
    """m = number of distinct gold operand cells."""
    return len(_as_cellset(gold))


def covered_gold_cells(gold: Iterable, retrieved_chunks: Iterable) -> List[Cell]:
    """Gold operand cells covered by >=1 retrieved chunk (``chunk.covers(r, c)``).

    Bridges chunk-based retrieval (``rag_agent.retrieve``) to the cell-set OSC
    metrics: feed the returned list as the ``retrieved`` argument of
    :func:`operand_set_completeness` / :func:`per_cell_recall`. Uses the same
    ``covers`` predicate as the repo's ``operand_recall`` for consistency.
    """
    return [
        (int(op.row), int(op.col))
        for op in gold
        if any(ch.covers(op.row, op.col) for ch in retrieved_chunks)
    ]


def header_depth(top_paths: Sequence[Sequence[str]],
                 left_paths: Sequence[Sequence[str]]) -> int:
    """d = deepest header-path length across the table's top and left trees."""
    def _maxlen(paths: Sequence[Sequence[str]]) -> int:
        return max((len(p) for p in paths), default=0)
    return max(_maxlen(top_paths), _maxlen(left_paths))


# --- stratification bins ---------------------------------------------------

def bin_scope(m: int) -> str:
    if m <= 1:
        return "1"
    if m == 2:
        return "2"
    if m <= 4:
        return "3-4"
    if m <= 8:
        return "5-8"
    return "9+"


def bin_depth(d: int) -> str:
    if d <= 1:
        return "flat(d<=1)"
    if d == 2:
        return "d2"
    return "d3+"


# --- per-query record + aggregation ---------------------------------------

def evaluate_query(gold: Iterable, retrieved: Iterable,
                   query_id: str = "",
                   header_depth_val: int = 0,
                   aggregation: str = "none") -> Dict:
    """Build a per-query record with OSC, per-cell recall and strata labels."""
    m = scope_size(gold)
    return {
        "query_id": query_id,
        "osc": operand_set_completeness(gold, retrieved),
        "per_cell_recall": per_cell_recall(gold, retrieved),
        "scope_size": m,
        "header_depth": int(header_depth_val),
        "aggregation": aggregation or "none",
    }


def aggregate_by(records: List[Dict], group_fn: Callable[[Dict], str]) -> Dict[str, Dict]:
    """Group records by ``group_fn`` and report mean OSC / per-cell recall / n."""
    groups: Dict[str, List[Dict]] = {}
    for rec in records:
        groups.setdefault(str(group_fn(rec)), []).append(rec)
    out: Dict[str, Dict] = {}
    for key, recs in sorted(groups.items()):
        out[key] = {
            "n": len(recs),
            "osc": mean(r["osc"] for r in recs),
            "per_cell_recall": mean(r["per_cell_recall"] for r in recs),
        }
    return out


def summarize(records: List[Dict]) -> Dict:
    """Overall + per-strata (scope size, depth, aggregation) OSC summary."""
    if not records:
        return {"overall": {"n": 0, "osc": 0.0, "per_cell_recall": 0.0}}
    return {
        "overall": {
            "n": len(records),
            "osc": mean(r["osc"] for r in records),
            "per_cell_recall": mean(r["per_cell_recall"] for r in records),
        },
        "by_scope": aggregate_by(records, lambda r: bin_scope(r["scope_size"])),
        "by_depth": aggregate_by(records, lambda r: bin_depth(r["header_depth"])),
        "by_aggregation": aggregate_by(records, lambda r: r["aggregation"]),
    }
