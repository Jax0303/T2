# SPDX-License-Identifier: MIT
"""v3.2 코드 변경 뒤 보존 확인과 v3.1 → v3.2 차이 — 모델 없음 (PREREG-2026-09-14-header-v3.md, 정정 3).

  1. v3.1 다섯 arm·v3 본 방법의 색인 문장 해시가 변경 전(preserve_before.json)과 같은가.
  2. v3.1 → v3.2 에서 문장이 바뀐 표·셀·행. 바뀐 행 중 v2 행 경로로 돌아간 행과 v2·v3.1 어느 쪽과도 다른 행.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_2_preserve.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analysis.header_v3_impact import ARMS, cells, corpus                       # noqa: E402
from mh_arms import build_tables, load_population                              # noqa: E402
from rag_agent.eval.artifacts import digest, provenance                        # noqa: E402

D = ROOT / "results/mh_header_v3_2"
SEED = 20260914


def main() -> int:
    out_path = D / "preserve_after.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — 덮어쓰지 않는다")
    before = json.loads((D / "preserve_before.json").read_text(encoding="utf-8"))["digests"]
    _, docs, _ = load_population("train")
    now = {}
    t3 = build_tables(docs, "v3")[0]
    texts = corpus(t3, "cell")[0]
    now["v3/cell"] = {"sha256": digest(texts), "n_units": len(texts)}
    del t3, texts
    t31 = build_tables(docs, "v3.1")[0]
    for arm in ARMS:
        texts = corpus(t31, arm)[0]
        now[f"v3.1/{arm}"] = {"sha256": digest(texts), "n_units": len(texts)}
    c2, c31 = cells(build_tables(docs, "v2")[0]), cells(t31)
    t32 = build_tables(docs, "v3.2")[0]
    c32 = cells(t32)
    changed = sorted(k for k in set(c31) & set(c32) if c31[k][3] != c32[k][3])
    rows = {}
    for k in changed:
        rows.setdefault((k[0], k[1]), k)
    back = {row for row, k in rows.items() if k in c2 and c32[k][0] == c2[k][0]}
    other = sorted(set(rows) - back)
    report = {
        "prereg": "PREREG-2026-09-14-header-v3.md 정정 3", "models_run": False,
        "hash_same_as_before_change": {k: now[k] == before[k] for k in before}, "digests_after": now,
        "v3_1_to_v3_2": {
            "cells_only_in_one": len(set(c31) ^ set(c32)),
            "tables": len({k[0] for k in changed}), "cells": len(changed), "rows": len(rows),
            "cells_col_path_changed": sum(c31[k][1] != c32[k][1] for k in changed),
            "rows_back_to_v2_row_path": len(back), "rows_unlike_v2_and_v3_1": len(other),
            "examples_unlike_v2_and_v3_1": [
                {"cell": list(rows[r]), "v2": list(c2[rows[r]][0]) if rows[r] in c2 else None,
                 "v3.1": list(c31[rows[r]][0]), "v3.2": list(c32[rows[r]][0])}
                for r in random.Random(SEED).sample(other, min(6, len(other)))]},
        "provenance": provenance(ROOT)}
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
