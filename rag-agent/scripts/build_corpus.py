#!/usr/bin/env python3
"""Phase 1 — HiTab → 검색 코퍼스 재구성.

전체 표(3597개)를 단일 코퍼스로 모으고, 원본 question→gold_table 정렬을
검색 정답으로 그대로 사용한다(재배열, 답 변경 없음). TARGET(Ji et al. 2025)이
Spider/BIRD/FeTaQA를 검색 태스크로 재구성한 표준 절차와 동일.

재발명 금지: 직렬화/헤더트리는 기존 T2 구현(src/serializers, src/data/header_tree)
을 그대로 재사용해 prebuilt chroma 임베딩과 직렬화가 일치하도록 한다.

산출:
  corpus/tables.jsonl                       {table_id, title, n_rows, n_cols,
                                             top_header_depth, left_header_depth, raw_cells}
  queries.jsonl                             {query_id, question, gold_table_id, answer,
                                             aggregation_label, split}
  corpus/serialized/{fmt}/records.jsonl     {table_id, [chunk_id,] text}
  logs/exclusions.jsonl                     제외 건 + 사유

사용: python scripts/build_corpus.py [--data-dir data/hitab] [--out-dir .]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- 기존 T2 구현 재사용 (재발명 금지) ---
T2_SRC = "/home/user/T2/hart-table-retrieval"
if T2_SRC not in sys.path:
    sys.path.insert(0, T2_SRC)

from src.data.header_tree import HeaderTree  # noqa: E402
from src.serializers.plain_markdown import PlainMarkdownSerializer  # noqa: E402
from src.serializers.json_kv import JsonKeyValueSerializer  # noqa: E402
from src.serializers.header_path import HeaderPathSerializer  # noqa: E402

# repo loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_agent.data.loader import load_samples, load_table  # noqa: E402

SPLITS = ["train", "dev", "test"]
SERIALIZERS = {
    "plain_markdown": PlainMarkdownSerializer(),
    "json_kv": JsonKeyValueSerializer(),
    "header_path": HeaderPathSerializer(),
}


def tree_depth(paths):
    return max((len(p) for p in paths), default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "corpus" / "serialized").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    excl = open(out / "logs" / "exclusions.jsonl", "w", encoding="utf-8")

    def exclude(kind, ident, reason):
        excl.write(json.dumps({"kind": kind, "id": ident, "reason": reason},
                              ensure_ascii=False) + "\n")

    # 1) 전체 표 id 수집 (파일 기준, 기계적)
    tables_root = Path(args.data_dir) / "data" / "tables"
    table_ids = set()
    for sub in ("hmt", "raw"):
        d = tables_root / sub
        if d.exists():
            for p in d.glob("*.json"):
                table_ids.add(p.stem)
    table_ids = sorted(table_ids)
    print(f"[corpus] unique table files: {len(table_ids)}")

    # 2) 표 → tables.jsonl + 직렬화
    ser_files = {name: open(out / "corpus" / "serialized" / f"{name}.records.jsonl",
                            "w", encoding="utf-8") for name in SERIALIZERS}
    f_tables = open(out / "corpus" / "tables.jsonl", "w", encoding="utf-8")
    kept_ids = set()
    n_chunks = {name: 0 for name in SERIALIZERS}

    for tid in table_ids:
        table = load_table(tid, args.data_dir)
        if table is None:
            exclude("table", tid, "load_table returned None")
            continue
        try:
            tree = HeaderTree()
            tree.build_tree(table)
        except Exception as e:  # noqa: BLE001
            exclude("table", tid, f"header_tree build failed: {e}")
            continue

        data = table.get("data", [])
        n_rows = len(data)
        n_cols = len(data[0]) if data else 0
        rec = {
            "table_id": tid,
            "title": table.get("title", ""),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "top_header_depth": tree_depth(tree.get_top_paths()),
            "left_header_depth": tree_depth(tree.get_left_paths()),
            # 원본 계층 구조 보존
            "raw_cells": {
                "top_root": table.get("top_root"),
                "left_root": table.get("left_root"),
                "data": data,
            },
        }
        f_tables.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 3 직렬화 동시 생성
        for name, ser in SERIALIZERS.items():
            try:
                results = ser.serialize(table, tree)
            except Exception as e:  # noqa: BLE001
                exclude("serialize", f"{tid}:{name}", str(e))
                continue
            for i, (text, _meta) in enumerate(results):
                row = {"table_id": tid, "text": text}
                if name == "header_path":
                    row["chunk_id"] = f"{tid}#{i}"
                ser_files[name].write(json.dumps(row, ensure_ascii=False) + "\n")
                n_chunks[name] += 1
        kept_ids.add(tid)

    f_tables.close()
    for f in ser_files.values():
        f.close()
    print(f"[corpus] tables kept: {len(kept_ids)}  serialization chunks: {n_chunks}")

    # 3) queries.jsonl (원본 정렬 그대로) + orphan 검증
    f_q = open(out / "queries.jsonl", "w", encoding="utf-8")
    n_q = {s: 0 for s in SPLITS}
    orphans = 0
    for split in SPLITS:
        try:
            samples = load_samples(args.data_dir, split)
        except FileNotFoundError:
            exclude("split", split, "samples file missing")
            continue
        for s in samples:
            gid = s.get("table_id")
            if gid not in kept_ids:
                orphans += 1
                exclude("query", s.get("id", "?"),
                        f"gold_table_id {gid} not in corpus (split={split})")
                continue
            f_q.write(json.dumps({
                "query_id": s.get("id"),
                "question": s.get("question"),
                "gold_table_id": gid,
                "answer": s.get("answer"),
                "aggregation_label": s.get("aggregation"),
                "split": split,
            }, ensure_ascii=False) + "\n")
            n_q[split] += 1
    f_q.close()
    excl.close()

    print(f"[queries] per split: {n_q}  total={sum(n_q.values())}  orphans={orphans}")
    # 검증 게이트 1 요약
    summary = {
        "n_tables_files": len(table_ids),
        "n_tables_kept": len(kept_ids),
        "n_serialization_chunks": n_chunks,
        "n_queries": n_q,
        "n_queries_total": sum(n_q.values()),
        "orphans": orphans,
        "len_corpus_eq_unique_table_ids": len(kept_ids) == len(table_ids),
    }
    (out / "corpus" / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("[gate1]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
