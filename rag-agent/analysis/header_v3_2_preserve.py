# SPDX-License-Identifier: MIT
"""헤더 규칙 코드 변경 뒤 보존 확인과 이전 규칙 → 새 규칙 차이 — 모델 없음 (PREREG-2026-09-14-header-v3.md).

  1. `<dir>/preserve_before.json` 에 적힌 규칙·arm 의 색인 문장 해시가 변경 전과 같은가.
  2. `--base` → `--new` 에서 문장이 바뀐 표·셀·행. 바뀐 행 중 v2 행 경로로 돌아간 행과 v2·base 어느 쪽과도 다른 행.

  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_2_preserve.py                    # 정정 3 (v3.1→v3.2)
  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python analysis/header_v3_2_preserve.py \
      --dir results/mh_header_v3_3 --base v3.2 --new v3.3                                            # 정정 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analysis.header_v3_impact import cells, corpus                            # noqa: E402
from mh_arms import build_tables, load_population                              # noqa: E402
from rag_agent.eval.artifacts import digest, provenance                        # noqa: E402

SEED = 20260914


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/mh_header_v3_2")
    ap.add_argument("--base", default="v3.1")
    ap.add_argument("--new", default="v3.2")
    a = ap.parse_args()
    D = ROOT / a.dir
    out_path = D / "preserve_after.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — 덮어쓰지 않는다")
    before = json.loads((D / "preserve_before.json").read_text(encoding="utf-8"))["digests"]
    _, docs, _ = load_population("train")
    now, built = {}, {}
    for rule in dict.fromkeys(k.split("/")[0] for k in before):
        tabs = build_tables(docs, rule)[0]
        for key in (k for k in before if k.split("/")[0] == rule):
            texts = corpus(tabs, key.split("/")[1])[0]
            now[key] = {"sha256": digest(texts), "n_units": len(texts)}
        if rule == a.base:
            built[rule] = tabs
        del tabs
    c2 = cells(build_tables(docs, "v2")[0])
    cb = cells(built.get(a.base) or build_tables(docs, a.base)[0])
    cn = cells(build_tables(docs, a.new)[0])
    changed = sorted(k for k in set(cb) & set(cn) if cb[k][3] != cn[k][3])
    rows = {}
    for k in changed:
        rows.setdefault((k[0], k[1]), k)
    back = {row for row, k in rows.items() if k in c2 and cn[k][0] == c2[k][0]}
    other = sorted(set(rows) - back)
    b_ = a.base.replace(".", "_")
    report = {
        "prereg": "PREREG-2026-09-14-header-v3.md", "models_run": False,
        "hash_same_as_before_change": {k: now[k] == before[k] for k in before}, "digests_after": now,
        f"{b_}_to_{a.new.replace('.', '_')}": {
            "cells_only_in_one": len(set(cb) ^ set(cn)),
            "tables": len({k[0] for k in changed}), "cells": len(changed), "rows": len(rows),
            "cells_col_path_changed": sum(cb[k][1] != cn[k][1] for k in changed),
            "rows_back_to_v2_row_path": len(back), f"rows_unlike_v2_and_{b_}": len(other),
            f"examples_unlike_v2_and_{b_}": [
                {"cell": list(rows[r]), "v2": list(c2[rows[r]][0]) if rows[r] in c2 else None,
                 a.base: list(cb[rows[r]][0]), a.new: list(cn[rows[r]][0])}
                for r in random.Random(SEED).sample(other, min(6, len(other)))]},
        "provenance": provenance(ROOT)}
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
