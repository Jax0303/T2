#!/usr/bin/env python3
"""진단 Phase 1 — OpenWikiTable(flat) → 표 단위 검색 코퍼스 재구성.

목적: flat↔hierarchical 복잡도 축의 *flat* 절반을 HiTab과 동일 스키마로 맞춘다.
질문의 gold table은 `original_table_id`로 매핑(표 단위 검색; 청크 단위 아님).
TARGET(Ji et al. 2025)이 기존 QA 데이터셋을 검색 태스크로 재구성한 절차와 동일.

산출:
  diag/flat/tables.jsonl   {table_id, page_title, section_title, caption,
                            header, rows, n_rows, n_cols, dataset}
  diag/flat/queries.jsonl  {query_id, question, gold_table_id, answer, split}
  diag/flat/build_summary.json

사용: python scripts/diag_build_flat.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

SRC = Path("data/openwikitable/repo/data")
OUT = Path("diag/flat")


def load_tables():
    t = json.load(open(SRC / "tables.json"))
    cols = list(t.keys())
    n = len(t["original_table_id"])
    tables = {}
    for i in range(n):
        k = str(i)
        rec = {c: t[c][k] for c in cols}
        tid = rec["original_table_id"]
        tables[tid] = {
            "table_id": tid,
            "page_title": rec.get("page_title") or "",
            "section_title": rec.get("section_title") or "",
            "caption": rec.get("caption") or "",
            "header": rec.get("header") or [],
            "rows": rec.get("rows") or [],
            "n_rows": len(rec.get("rows") or []),
            "n_cols": len(rec.get("header") or []),
            "dataset": rec.get("dataset") or "",
        }
    return tables


def load_queries(split, fname):
    q = json.load(open(SRC / fname))
    n = len(q["question_id"])
    out = []
    for i in range(n):
        k = str(i)
        out.append({
            "query_id": q["question_id"][k],
            "question": q["question"][k],
            "gold_table_id": q["original_table_id"][k],
            "answer": q["answer"][k],
            "split": split,
        })
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    print(f"[flat] tables: {len(tables)}")
    with open(OUT / "tables.jsonl", "w", encoding="utf-8") as f:
        for rec in tables.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    splits = [("train", "train.json"), ("dev", "valid.json"), ("test", "test.json")]
    all_q = []
    orphans = 0
    for split, fn in splits:
        qs = load_queries(split, fn)
        for r in qs:
            if r["gold_table_id"] not in tables:
                orphans += 1
        all_q.extend(qs)
        print(f"[flat] {split}: {len(qs)} queries")
    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as f:
        for r in all_q:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_rows = [t["n_rows"] for t in tables.values()]
    n_cols = [t["n_cols"] for t in tables.values()]
    summary = {
        "n_tables": len(tables),
        "n_queries_total": len(all_q),
        "by_split": {s: sum(1 for r in all_q if r["split"] == s) for s, _ in splits},
        "orphans": orphans,
        "n_rows_median": sorted(n_rows)[len(n_rows) // 2],
        "n_cols_median": sorted(n_cols)[len(n_cols) // 2],
        "complexity": "flat (single header row)",
    }
    json.dump(summary, open(OUT / "build_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    assert orphans == 0, f"orphan gold tables: {orphans}"
    print("[flat] orphan=0 OK")


def _no_args() -> None:
    """This script takes no options. Without a parser, argparse-style flags are
    silently ignored and the full experiment runs anyway — which is how a bare
    ``--help`` sweep silently regenerated committed artifacts."""
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _no_args()
    sys.exit(main())
