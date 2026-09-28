"""PREREG-2026-09-28-mh-name-append.md: 정답 셀 1개 문항(조회 212, 산술 71)과 질문 뒤에 붙일 말(정답 셀의 열 경로) 만들기.

실행:  PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/mh_name_append_20260928/make_suffix.py
출력:  이 폴더의 uids.json, suffix.json, groups.json
그룹: 값 비교 조회(results/lookup_vs_arith_20260928/why_rank1.py 의 VC 단어 목록), 그 밖의 조회, 산술.
붙일 말: 최종 머리글 규칙(v3.3u) 표에서 정답 셀의 열 경로. 식별자('Table k', 'row N', 'column N')는 뺀다.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, "scripts")
import mh_arms as ma  # noqa: E402

src = Path("results/lookup_vs_arith_20260928/why_rank1.py").read_text(encoding="utf-8")
VC = eval("re.compile(" + re.search(r"VC = re\.compile\((.*?)\)\n", src, re.S).group(1) + ")")
IDENT = re.compile(r"^(Table \d+|row \d+|column \d+)$")

queries, docs, _ = ma.load_population("train")
tables, hdr = ma.build_tables(docs, "v3.3u", "none")
live = {(tid, i, j) for tid, tab in tables.items() for i, rr in enumerate(tab.table.data) for j, v in enumerate(rr) if str(v).strip()}
qs = [q for q in ma.resolve_gold(queries, tables, hdr, live) if not q["excluded"] and len(q["gold"]) == 1]
groups, suffix = {}, {}
for q in qs:
    (tid, i, j), = q["gold"]
    groups[q["uid"]] = "산술" if q["kind"] == "arith" else ("값 비교 조회" if VC.search(q["question"]) else "그 밖의 조회")
    suffix[q["uid"]] = " ".join(x for x in tables[tid].table.col_path(j) if not IDENT.match(x))
from collections import Counter  # noqa: E402
n = Counter(groups.values())
assert n == {"값 비교 조회": 141, "그 밖의 조회": 71, "산술": 71}, n
(HERE / "uids.json").write_text(json.dumps(sorted(groups), indent=0) + "\n")
(HERE / "suffix.json").write_text(json.dumps(suffix, ensure_ascii=False, indent=0) + "\n")
(HERE / "groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=0) + "\n")
print(dict(n), "빈 붙일 말:", sum(not v for v in suffix.values()))
