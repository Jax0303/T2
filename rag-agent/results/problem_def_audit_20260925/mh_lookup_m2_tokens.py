"""MultiHiertt cap300 조회 셀2+ 그룹(300건): 표 전체 대 본 방법(cell_uniq) 리더 입력 토큰(n_tok). 파일 읽기만."""
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = "results/mh_arms/cap300_20260924/"
out = {}
for arm in ("fulltable", "cell_uniq"):
    rs = [r for r in map(json.loads, open(ROOT / D / f"{arm}.jsonl")) if r["layer"] == "lookup_m2+"]
    t = [r["n_tok"] for r in rs]
    out[arm] = {"file": D + f"{arm}.jsonl (layer=lookup_m2+, n_tok)", "query_count": len(rs),
                "mean": round(st.mean(t), 1), "median": st.median(t), "max": max(t),
                "answer_correct": sum(r["answer_correct"] for r in rs)}
print(json.dumps(out, indent=1, ensure_ascii=False))
