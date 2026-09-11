#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Public report entrypoint. Explicit manifests replace implicit legacy file selection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.validated_tables import DEFAULT_MANIFEST, build, main
from rag_agent.eval.artifacts import read_records
import json
D = ROOT / "results/retrieval_accuracy"
# Compatibility for historical audits; these are not full-system baselines.
_spec = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
BASELINE_ROWS = [(a["tag"], a["label"], "", a["role"]) for a in _spec["arms"]]
BASELINE_ROWS.append(("t_table_hybrid", "표 통째 — 셀 선택 없는 통제", "", "control"))
NO_CELL_STEP = {"t_table_hybrid"}

def primary_ids(ref="t_s3c_hybrid"):
    return {q for q, r in read_records(D / f"{ref}_records.jsonl").items()
            if "correct" in r and r["mode"] == "all" and r["m"] == 1
            and (r.get("aggregation") or "none") == "none"}

if __name__ == "__main__":
    main()
