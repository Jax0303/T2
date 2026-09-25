"""GATE 0-2: 현재 작업 트리(미커밋)의 표 전체 대 본 방법 짝 — 파일 읽기와 정확 McNemar 만."""
import json
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
full = {r["query_id"]: r["correct_base"] for r in map(json.loads, open(ROOT / "results/fulltable_20260924/hitab_rows.jsonl"))}
ours = {r["query_id"]: r["correct_base"] for r in map(json.loads, open(ROOT / "results/fair_filter_20260921/rows.jsonl")) if r["arm"] == "ours"}
assert set(full) == set(ours)
b = sum(full[q] and not ours[q] for q in full); c = sum(ours[q] and not full[q] for q in full)
tok = sorted(r["reader_input_tokens"] for r in map(json.loads, open(ROOT / "results/fulltable_20260924/hitab_rows.jsonl")))
print(json.dumps({"hitab_300": {
    "fulltable_file": "results/fulltable_20260924/hitab_rows.jsonl",
    "ours_file": "results/fair_filter_20260921/rows.jsonl (arm=ours, correct_base; records t_sleaf_gold)",
    "query_count": len(full), "fulltable_correct": sum(full.values()), "ours_correct": sum(ours.values()),
    "fulltable_only": b, "ours_only": c, "mcnemar_exact_p": round(binomtest(b, b + c, .5).pvalue, 4),
    "fulltable_reader_input_tokens": {"min": tok[0], "median": (tok[149] + tok[150]) / 2, "max": tok[-1]}}},
    indent=1))
