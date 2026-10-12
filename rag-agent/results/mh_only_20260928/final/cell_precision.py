"""셀 정밀도 |G∩R_k|/|R_k| (문항 평균, k=1,5,10,20) — 표 4-5 셀P@k 열. 새 실행 없음.
실행: .venv/bin/python results/mh_only_20260928/final/cell_precision.py  (저장소 rag-agent 에서) → cell_precision.md"""
import json, statistics, sys
from pathlib import Path
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from rpm import units_of  # noqa: E402
from retrieval_table import METHODS, B, ids  # noqa: E402
s200 = set(ids(B + "ids200.jsonl"))
L = ["| 방법 | 범위 | 셀P@1 | 셀P@5 | 셀P@10 | 셀P@20 |", "|---|---|---:|---:|---:|---:|"]
for name, p in METHODS:
    per = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); u = units_of(r)
        if u is None or not r.get("gold_ids"):
            continue
        g = set(r["gold_ids"]); per[r["query_id"]] = {}
        for k in (1, 5, 10, 20):
            R = set().union(*u[:k]) if u[:k] else set()
            per[r["query_id"]][k] = len(g & R) / len(R) if R else 0.0
    for tag, keep in (("200", s200), ("2885", None)):
        q = [x for x in per if keep is None or x in keep]
        if keep is not None and len(q) != 200 or keep is None and len(q) != 2885:
            continue
        L.append(f"| {name} | {tag} | " + " | ".join(f"{100 * statistics.mean(per[x][k] for x in q):.1f}" for k in (1, 5, 10, 20)) + " |")
(HERE / "cell_precision.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
