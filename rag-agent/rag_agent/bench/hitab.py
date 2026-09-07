# SPDX-License-Identifier: MIT
"""HiTab -> unified schema adapter.

Tables and questions come from the local HiTab dump. Gold operands are the data
cells the question's answer is read from, resolved by
:mod:`rag_agent.bench.hitab_grid` -- ``answer_formulas`` + ``reference_cells_map``
mapped onto the data matrix through the two header trees.

What used to be here instead: a search over a 6x6 grid of (header_rows,
header_cols) offsets for one that made the annotated VALUES line up, reading
gold from ``quantity_link`` and only where the value parsed as a number. On the
test split that left 351 of 1,584 questions with no gold at all -- every
question whose answer is a label rather than a number ("which province had the
highest rate") -- and those questions were then dropped from the population by
whatever consumed this. The mapping is exact and does not need guessing; see
``hitab_grid``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..data.loader import load_samples, load_table
from ..stores.original_store import build_original_table, OriginalTable
from . import hitab_grid as _grid
from .schema import BenchTable, BenchQuery, GoldOperand

SOURCE = "hitab"


def _bench_table_from_original(ot: OriginalTable) -> BenchTable:
    return BenchTable(
        table_id=ot.table_id,
        title=ot.title,
        data=ot.data,
        top_paths=[ot.col_path(c) for c in range(ot.n_cols)],
        left_paths=[ot.row_path(r) for r in range(ot.n_rows)],
        source=SOURCE,
    )


def _operand(bt: BenchTable, r: int, c: int) -> GoldOperand:
    v = bt.cell(r, c)
    fv = _grid.norm_value(v)
    try:
        num = float(fv)
    except ValueError:
        num = None
    return GoldOperand(row=r, col=c, header_path=bt.full_path(r, c),
                       value=num, value_type="number" if num is not None else "string")


def load_tables(data_dir: str = "data/hitab", table_ids: Optional[set] = None) -> Dict[str, BenchTable]:
    """Load the requested tables (or all referenced by ``table_ids``) as BenchTables."""
    out: Dict[str, BenchTable] = {}
    if table_ids is None:
        return out
    for tid in table_ids:
        raw = load_table(tid, data_dir)
        if raw is None:
            continue
        out[tid] = _bench_table_from_original(build_original_table(raw))
    return out


def load_queries(
    data_dir: str = "data/hitab",
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> tuple:
    """Return ``(queries, tables)`` for the split, with gold operands resolved.

    Every question in the split comes back. One whose gold cannot be resolved
    comes back with an EMPTY ``gold_operands`` and the reason on
    ``exclusion_reason`` -- a caller that wants only scoreable questions filters
    on that and knows how many it dropped, instead of the count vanishing.
    """
    samples = load_samples(data_dir, split, max_samples)
    tables: Dict[str, BenchTable] = {}
    grids: Dict[str, object] = {}
    queries: List[BenchQuery] = []
    for s in samples:
        tid = s.get("table_id")
        if tid not in grids:
            g = _grid.load_table(tid, data_dir)
            grids[tid] = g
            if g is not None:
                tables[tid] = _bench_table_from_original(g.table)
        g = grids[tid]
        if g is None:
            continue
        bt = tables[tid]
        cells, mode, why = _grid.gold_target(s, g)
        ops = ([] if why else
               [_operand(bt, i, j) for _t, i, j in sorted(cells)]) if mode == "all" else []
        agg = s.get("aggregation")
        q = BenchQuery(
            query_id=s.get("id"),
            question=s.get("question", ""),
            gold_table_id=tid,
            answer=s.get("answer", []),
            gold_operands=ops,
            aggregation=agg[0] if isinstance(agg, list) and agg else agg,
            split=split,
            source=SOURCE,
            exclusion_reason=why or (None if mode == "all" else "gold_is_header_cell"),
        )
        queries.append(q)
    return queries, tables
