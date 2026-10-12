"""원고 3.3.1절 근거 — 머리글 판정 규칙과 ' > ' 기호. 새 모델 실행 없음(CPU, 표 파싱만).

1) HiTab 표 3,597개(data/hitab/data/tables/raw)에는 사람이 붙인 머리글 트리가 있다. 본 방법이 MultiHiertt 에 쓰는
   판정 절차(mh_arms.build_tables 와 같은 순서·같은 규칙: 행 먼저 열 1개 가정, rule="v3.1", 그 행 수로 열)를
   HiTab 원 격자에 그대로 적용해 머리글 행 수·열 수가 사람 표시와 같은 표의 비율을 센다.
   사람 표시 = 트리가 처음 가리키는 데이터 행/열 번호(tree_reconstruct_hitab_raw.py 와 같은 정의).
   주의: 규칙을 만들 때 HiTab 표도 참고했으므로 독립 검증은 아니다.
2) MultiHiertt 평가 문항 2,885개의 문서 표에서 첫째 신호(왼쪽 위 칸이 빔)가 쓰이는 비율, 머리글 행·열 수 분포,
   머리글 칸 글자에 후보 구분 기호가 나오는 칸 수, HTML 에 <th> 표시가 있는 표 수.
Run: PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/header_rule_check_20261001/header_rule_check.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import load_population                                           # noqa: E402
from tree_reconstruct_hitab_raw import tree_lines                              # noqa: E402
from rag_agent.reconstruct import guess_n_header_cols, guess_n_header_rows, parse_html_table  # noqa: E402
from rag_agent.reconstruct.header_grid import _left_region_blank               # noqa: E402


def judge(g):
    nhr = max(1, min(guess_n_header_rows(g, n_header_cols=1, rule="v3.1"), len(g) - 1))
    return nhr, max(1, min(guess_n_header_cols(g, n_header_rows=nhr), len(g[0]) - 1))


pct = lambda a, b: round(100 * a / b, 1)
out = {}

# 1) HiTab
h = Counter()
for p in sorted((ROOT / "data/hitab/data/tables/raw").glob("*.json")):
    d = json.load(open(p, encoding="utf-8"))
    g = [[str(x) if x is not None else "" for x in row] for row in (d.get("texts") or [])]
    cols_c, _ = tree_lines(d.get("top_root") or {}, "top")
    rows_c, _ = tree_lines(d.get("left_root") or {}, "left")
    if len(g) < 3 or len(g[0]) < 2 or not cols_c or not rows_c or min(rows_c) <= 0 or min(cols_c) <= 0:
        h["skipped"] += 1
        continue
    nhr, nhc = judge(g)
    corner = _left_region_blank(g, 0, 1)
    h["tables"] += 1
    h["row_ok"] += nhr == min(rows_c)
    h["col_ok"] += nhc == min(cols_c)
    h["both_ok"] += nhr == min(rows_c) and nhc == min(cols_c)
    h["corner_blank"] += corner
    h["row_ok_corner_blank"] += corner and nhr == min(rows_c)
out["hitab"] = {**h, "row_ok_pct": pct(h["row_ok"], h["tables"]), "col_ok_pct": pct(h["col_ok"], h["tables"]),
                "both_ok_pct": pct(h["both_ok"], h["tables"]),
                "row_ok_pct_corner_blank": pct(h["row_ok_corner_blank"], h["corner_blank"]),
                "row_ok_pct_corner_text": pct(h["row_ok"] - h["row_ok_corner_blank"], h["tables"] - h["corner_blank"])}

# 2) MultiHiertt — 평가 문항 2,885개(LangChain 청크 기록의 문항 = 본 방법 채점 모집단)의 문서
REC = ROOT / "results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl"
uids = {json.loads(l)["query_id"] for l in open(REC, encoding="utf-8")}
_, docs, _ = load_population("train")
assert uids <= set(docs), "기록의 문항 id 가 데이터셋 uid 와 다르다"
SEPS = [">", "→", ".", "/", "-", "|", ",", ":"]
m, sep = Counter(), Counter()
for uid in sorted(uids):
    for html in docs[uid][0]:
        g = parse_html_table(html)
        if len(g) < 3 or len(g[0]) < 2:                                          # build_tables 와 같은 제외
            continue
        nhr, nhc = judge(g)
        m["tables"] += 1
        m["th_tag"] += "<th" in html.lower()
        m["corner_blank"] += _left_region_blank(g, 0, 1)
        m[f"header_rows_{min(nhr, 4)}"] += 1
        m[f"header_cols_{min(nhc, 3)}"] += 1
        for r, row in enumerate(g):
            for j, x in enumerate(row):
                if (r < nhr or j < nhc) and x.strip():
                    sep["header_cells"] += 1
                    for s in SEPS:
                        sep[s] += s in x
out["multihiertt"] = {"docs": len(uids), **m, "corner_blank_pct": pct(m["corner_blank"], m["tables"]),
                      **{f"{k}_pct": pct(m[k], m["tables"]) for k in sorted(m) if k.startswith("header_")}}
out["multihiertt_header_cell_symbols"] = {"header_cells": sep["header_cells"],
                                          **{s: [sep[s], round(100 * sep[s] / sep["header_cells"], 3)] for s in SEPS}}
(Path(__file__).parent / "header_rule_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
