#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Language-independence test of total-row detection on IM-TQA (Chinese).

The keyword detector (`is_total_row`, English regex "total/overall/all") cannot
transfer to non-English tables; the structural detector
(`is_total_row_structural`) defines a total by its VALUE (a row that sums a row
group in its own section) and should fire regardless of language. IM-TQA
(Zheng et al., ACL 2023) provides the same tables with BOTH Chinese and
machine-translated English cell values — a paired-language corpus: run both
detectors on both language versions of identical grids and compare.

Expected signature of language independence: structural hits are (near-)
identical across languages, keyword hits collapse on the Chinese version.
Precision aid: structural hits are arithmetic by construction (a fired row
provably sums a sibling/child group within rel_tol); we also report how many
keyword-style total labels ("合计/总计/小计" — Chinese total markers, used here
for EVALUATION only, never by the detector) the structural detector recovers.

Headers are self-reconstructed (guess_n_header_rows + reconstruct_row_paths),
same as the MultiHiertt pipeline — no IM-TQA header-type annotations are used.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.reconstruct.header_grid import (guess_n_header_rows,
                                               reconstruct_row_paths)
from rag_agent.retrieve.header_enum import (is_total_row,
                                            is_total_row_structural)

# evaluation-only gold-ish markers for Chinese total rows (NOT used by detectors)
_ZH_TOTAL = re.compile(r"合\s*计|总\s*计|小\s*计|合计数|总额|总计数")
_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _to_float(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    s = s.strip("()")  # (1,234) accounting negatives kept simple: magnitude
    try:
        return float(s)
    except ValueError:
        return None


class GridTable:
    """Minimal header_enum-compatible view over an IM-TQA cell grid."""

    def __init__(self, grid, n_header_rows: int, n_header_cols: int = 1):
        self.body = [row[n_header_cols:] for row in grid[n_header_rows:]]
        paths = reconstruct_row_paths(grid, n_header_rows, n_header_cols)
        self.left = paths[:len(self.body)]

    @property
    def n_rows(self):
        return len(self.body)

    @property
    def n_cols(self):
        return len(self.body[0]) if self.body else 0

    def row_path(self, r):
        return self.left[r] if 0 <= r < len(self.left) else []

    def cell_num(self, r, c):
        if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
            return _to_float(self.body[r][c])
        return None


def grids_of(table: dict):
    """(chinese_grid, english_grid) from an IM-TQA table record."""
    out = []
    for key in ("chinese_cell_value_list", "english_cell_value_list"):
        vals = table[key]
        grid = [[str(vals[cid]).strip() for cid in row]
                for row in table["cell_ID_matrix"]]
        out.append(grid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/imtqa")
    ap.add_argument("--splits", default="dev_tables.json,test_tables.json")
    ap.add_argument("--out", default="results/imtqa_structural_detector.json")
    args = ap.parse_args()

    tables = []
    for split in args.splits.split(","):
        tables += json.loads((Path(args.data_dir) / split).read_text())

    per_type = {}
    agree = disagree = 0
    zh_label_rows = zh_label_hit_struct = zh_label_hit_kw = 0
    rows_zh = hits_struct_zh = hits_struct_en = hits_kw_zh = hits_kw_en = 0
    n_tables = 0
    for t in tables:
        zh_grid, en_grid = grids_of(t)
        if len(zh_grid) < 3 or len(zh_grid[0]) < 2:
            continue
        nhr = max(1, min(guess_n_header_rows(zh_grid, n_header_cols=1),
                         len(zh_grid) - 1))
        zh, en = GridTable(zh_grid, nhr), GridTable(en_grid, nhr)
        if zh.n_rows == 0 or zh.n_cols == 0:
            continue
        n_tables += 1
        ttype = t.get("table_type", "?")
        st = per_type.setdefault(ttype, {"tables": 0, "struct_zh": 0,
                                         "struct_en": 0, "kw_zh": 0, "kw_en": 0})
        st["tables"] += 1
        for r in range(zh.n_rows):
            rows_zh += 1
            s_zh = is_total_row_structural(zh, r)
            s_en = is_total_row_structural(en, r)
            # keyword detector treats EMPTY row paths as total-like (a HiTab
            # convention); exclude those rows from the keyword count so the
            # language comparison isolates the regex itself.
            k_zh = bool(zh.row_path(r)) and is_total_row(zh, r)
            k_en = bool(en.row_path(r)) and is_total_row(en, r)
            hits_struct_zh += s_zh
            hits_struct_en += s_en
            hits_kw_zh += k_zh
            hits_kw_en += k_en
            st["struct_zh"] += s_zh
            st["struct_en"] += s_en
            st["kw_zh"] += k_zh
            st["kw_en"] += k_en
            agree += (s_zh == s_en)
            disagree += (s_zh != s_en)
            path = zh.row_path(r)
            if path and _ZH_TOTAL.search(str(path[-1])):
                zh_label_rows += 1
                zh_label_hit_struct += s_zh
                zh_label_hit_kw += k_zh

    report = {
        "corpus": {"tables": n_tables, "body_rows": rows_zh,
                   "splits": args.splits},
        "detector_hits": {
            "structural_on_chinese": hits_struct_zh,
            "structural_on_english": hits_struct_en,
            "keyword_on_chinese": hits_kw_zh,
            "keyword_on_english": hits_kw_en,
        },
        "structural_cross_language_agreement": {
            "agree_rows": agree, "disagree_rows": disagree,
            "rate": round(agree / max(1, agree + disagree), 4)},
        "labeled_zh_total_rows": {
            "n(rows whose own label matches 合计/总计/小计 etc.)": zh_label_rows,
            "recovered_by_structural": zh_label_hit_struct,
            "recovered_by_keyword_regex": zh_label_hit_kw,
            "structural_recall_on_labeled": round(
                zh_label_hit_struct / zh_label_rows, 4) if zh_label_rows else None,
        },
        "by_table_type": per_type,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
