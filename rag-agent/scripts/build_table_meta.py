#!/usr/bin/env python3
"""교수님 방식: 각 표의 구조 메타데이터(depth + 헤더-셀 경로)를 *미리 계산해 저장*.

추론 때 즉석 추출(cap 잘림)하는 대신, 여기서 한 번 전체를 정제해 디스크에 라벨링한다.
산출: data/table_meta/<bench>/<table_id>.json

메타 스키마:
  table_id, bench, n_rows, n_cols
  row_depth, col_depth           표 계층 깊이(경로 최대 길이)
  row_paths, col_paths           고유 헤더경로 전체 (cap 없음)  "a > b" 형태
  cell_paths                     각 데이터 셀의 풀경로 (row_path + col_path), 값 포함
사용: .venv/bin/python scripts/build_table_meta.py --benches hitab,finqa,wikisql
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def safe_name(tid: str) -> str:
    """table_id를 안전한 파일명으로 (슬래시 등 제거)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(tid))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import registry                       # noqa: E402
from scripts.binding_eval import bench_to_original         # noqa: E402


def build_meta(bench: str, bt) -> dict:
    ot = bench_to_original(bt)
    row_paths, col_paths = [], []
    seenr, seenc = set(), set()
    rdep = cdep = 0
    for r in range(ot.n_rows):
        p = ot.row_path(r)
        rdep = max(rdep, len(p))
        s = " > ".join(p)
        if s and s not in seenr:
            seenr.add(s); row_paths.append(s)
    for c in range(ot.n_cols):
        p = ot.col_path(c)
        cdep = max(cdep, len(p))
        s = " > ".join(p)
        if s and s not in seenc:
            seenc.add(s); col_paths.append(s)
    cell_paths = []
    for r in range(ot.n_rows):
        for c in range(ot.n_cols):
            fp = bt.full_path(r, c)
            if fp:
                cell_paths.append({"row": r, "col": c, "path": " > ".join(fp),
                                   "value": ot.cell(r, c)})
    return {"table_id": bt.table_id, "bench": bench,
            "n_rows": ot.n_rows, "n_cols": ot.n_cols,
            "row_depth": rdep, "col_depth": cdep,
            "row_paths": row_paths, "col_paths": col_paths,
            "cell_paths": cell_paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="hitab,finqa,wikisql")
    ap.add_argument("--out-dir", default="data/table_meta")
    args = ap.parse_args()

    for bench in [b.strip() for b in args.benches.split(",") if b.strip()]:
        _, tables = registry.load(bench, max_samples=None)
        outdir = ROOT / args.out_dir / bench
        outdir.mkdir(parents=True, exist_ok=True)
        n = 0
        depths = []
        for tid, bt in tables.items():
            if not bt.data:
                continue
            meta = build_meta(bench, bt)
            depths.append(meta["row_depth"])
            (outdir / f"{safe_name(tid)}.json").write_text(json.dumps(meta, ensure_ascii=False))
            n += 1
        maxd = max(depths) if depths else 0
        hier = sum(1 for d in depths if d >= 2)
        print(f"[{bench}] saved {n} tables → {outdir}  (row_depth max={maxd}, "
              f"hierarchical(depth>=2)={hier})", flush=True)


if __name__ == "__main__":
    main()
