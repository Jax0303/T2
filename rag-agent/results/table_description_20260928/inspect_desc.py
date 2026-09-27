#!/usr/bin/env python3
"""MultiHiertt table_description 확인 (실행 없음 — 임베딩·모델을 부르지 않는다). 2026-09-28.

모집단 = 본 방법 MultiHiertt 주 결과와 같은 train 표 근거 질의 2,908건의 문서(mh_arms.load_population).
본 방법 셀 문장 = mh_arms.build_tables(header_rule="v3.3u") + build_corpus(unit="cell", template="s3c"),
주 결과(rerun_20260926/mh/mh_train_s3c.json)의 corpus_text_sha256 과 같은지 먼저 확인한다.

짝짓기 키: table_description 키 "{표}-{행}-{열}" = 펼친 격자 좌표 = mh_arms.cell_id (table_evidence 와 같은 규약).

  PYTHONPATH=. .venv/bin/python results/table_description_20260928/inspect_desc.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from mh_arms import build_tables, load_population                    # noqa: E402
from retrieval_accuracy import build_corpus                          # noqa: E402
from rag_agent.eval.artifacts import digest                           # noqa: E402
from rag_agent.reconstruct.header_grid import parse_html_table_layout  # noqa: E402

OUT = Path(__file__).resolve().parent
MAIN_SHA = json.loads((ROOT / "results/rerun_20260926/mh/mh_train_s3c.json").read_text())["corpus_text_sha256"]
SEED = 20260928
_DESC = re.compile(r"^Table (\d+) shows (.*) is (.*?)\s*\.?\s*$", re.S)
norm = lambda s: re.sub(r"[,\s]", "", s or "")

queries, docs, _ = load_population("train")
tables, hdr = build_tables(docs, "v3.3u")
texts, covers, _, unit_tids, _ = build_corpus(
    "", sorted(tables), "s3c", "cell", {}, load=lambda tid, _d: tables.get(tid))
assert digest(texts) == MAIN_SHA, "본 방법 셀 문장이 주 결과와 다르다"
desc = {u: json.loads(d[1]) if isinstance(d[1], str) else d[1] for u, d in docs.items()}

# 1) 전체 셀: 모집단 문서의 모든 표(파싱 못 한 표 포함), 격자의 비어 있지 않은 칸 = 머리글 + 데이터.
c = Counter()
for uid, (htmls, _, _) in docs.items():
    keys = set(desc[uid])
    for t, html in enumerate(htmls):
        grid, _cov = parse_html_table_layout(html)
        tid = f"{uid}::{t}"
        for r, row in enumerate(grid):
            for col, v in enumerate(row):
                if not (v or "").strip():
                    continue
                has = f"{t}-{r}-{col}" in keys
                part = ("data" if tid in hdr and r >= hdr[tid][0] and col >= hdr[tid][1] else
                        "header" if tid in hdr else "unparsed_table")
                c[f"grid_{part}"] += 1
                c[f"grid_{part}_with_desc"] += has
    # table_description 쪽에서: 키가 어디에 떨어지는가
    for k in keys:
        t, r, col = map(int, k.split("-"))
        tid = f"{uid}::{t}"
        if tid not in hdr:
            c["desc_on_unparsed_table"] += 1
        elif r < hdr[tid][0] or col < hdr[tid][1]:
            c["desc_on_header"] += 1
        else:
            c["desc_on_data_position"] += 1
        c["desc_total"] += 1

# 2) 본 방법 색인 셀과 짝짓기
pairs = {}          # 색인 위치 -> (desc 문장, 값 일치)
for p, cov in enumerate(covers):
    (tid, i, j), = cov
    uid, t = tid.split("::")
    s = desc[uid].get(f"{t}-{i + hdr[tid][0]}-{j + hdr[tid][1]}")
    c["index_cells"] += 1
    if s is None:
        continue
    m = _DESC.match(s)
    same = bool(m) and norm(m.group(3)) == norm(tables[tid].table.data[i][j])
    pairs[p] = (s, same)
    c["index_with_desc"] += 1
    c["index_with_desc_value_match"] += same

# 3) 머리글 2층 이상(본 방법 복원 nhr >= 2) 표 3개 × 셀 5개, 시드 고정 무작위.
rng = random.Random(SEED)
by_tid = {}
for p, tid in enumerate(unit_tids):
    by_tid.setdefault(tid, []).append(p)
multi = sorted(t for t in by_tid if hdr[t][0] >= 2 and len(by_tid[t]) >= 5)
examples = []
for tid in rng.sample(multi, 3):
    for p in sorted(rng.sample(by_tid[tid], 5)):
        (_, i, j), = covers[p]
        examples.append({"table": tid, "header_rows": hdr[tid][0],
                         "cell": f"{tid.split('::')[1]}-{i + hdr[tid][0]}-{j + hdr[tid][1]}",
                         "ours_s3c": texts[p],
                         "table_description": pairs.get(p, (None,))[0]})

r = lambda a, b: round(a / b, 4)
stats = {
    "population": f"train table-evidence queries {len(queries)}, docs {len(docs)}",
    "counts": dict(sorted(c.items())),
    "grid_nonempty_cells_with_desc": r(sum(c[f"grid_{x}_with_desc"] for x in ("data", "header", "unparsed_table")),
                                       sum(c[f"grid_{x}"] for x in ("data", "header", "unparsed_table"))),
    "grid_data_cells_with_desc": r(c["grid_data_with_desc"], c["grid_data"]),
    "index_cells_pairable": r(c["index_with_desc"], c["index_cells"]),
    "index_cells_pairable_value_match": r(c["index_with_desc_value_match"], c["index_cells"]),
    "desc_sentences_on_index_cell": r(c["index_with_desc"], c["desc_total"]),
    "multi_header_tables": len(multi), "seed": SEED,
}
(OUT / "inspect_desc.json").write_text(json.dumps({"stats": stats, "examples": examples},
                                                  indent=1, ensure_ascii=False))
print(json.dumps(stats, indent=1, ensure_ascii=False))
for e in examples:
    print(f"\n[{e['table']} 머리글 {e['header_rows']}행 · 셀 {e['cell']}]\n  본 방법: {e['ours_s3c']}\n  desc  : {e['table_description']}")
