"""2026-09-26 HiTab e(s3c, 표 제목 접두어) 표 538개 색인, 단일 셀 991 중 틀린 85건 분해 (리더 없음, 재검색 없음 — records 만).
틀림 = t_s3c_split_labelabl_type_accuracy.jsonl 의 query_type == single_cell, retrieval_success == 0
(labelmix_20260926/analyze.py 의 split e 906 과 같은 판정).
1) 정답 표를 찾음 = 20셀(retrieved_cell_ids) 중 하나라도 정답 셀과 같은 table_id. records 의 gold_table_in_context 와 대조.
2) 정답 표 못 찾음 -> gold_table_missed.csv. 표 제목 = 색인 문장에 들어간 제목
   (with_page_title(표 제목, ToTTo 쪽 제목)); 1위 셀 = records context_units[0] 의 셀(검색 순위 순서.
   retrieved_cell_ids·context_cells 는 정렬된 목록이라 순위 순서가 아님). 같은 제목 = 문자열 일치.
   1위 셀 제목이 context_units[0] 문장의 "In the table '<제목>'" 과 같은지 대조(제목 빈 표 5개는 그 머리가 없음을 대조).
3) 정답 표 찾음 -> gold_table_found.csv. 정답 셀 순위 = 20위 밖은 21 로 자른 값, 그리고 records 의 gold_rank
   (전체 순위 500위까지, 500위 밖은 None).
4) 판정용 judge40.csv = 2) 의 40건. 셀 경로 = 행 머리글 경로 > 열 머리글 경로(' > ' 로 이음).
   1위 셀 문장이 context_units[0] 문장과 바이트 같은지(cell_unit, s3c) 대조.
   같은 제목 test 표 수 = records 의 table_id 538개 중 색인 제목이 정답 표 제목과 문자열 일치하는 표 수(정답 표 포함).
   판정 열 (가)(나)(다)는 비워 둔다. 정답 값 = records answer.
   정답 표·1위 표 전체 = judge40_tables/<번호>_<query_id>_{gold,top1}.md (markdown_source, 즉 표 전체 비교군이
   리더에 준 것과 같은 글에 첫 행 뒤 구분선 한 줄만 더함). CSV 에는 그 파일의 상대 경로.
실행: .venv/bin/python results/hitab_e_miss_20260926/analyze.py   (rag-agent 에서)
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import retrieval_accuracy as ra                                       # noqa: E402
from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.serialization.caption import with_page_title          # noqa: E402
from rag_agent.serialization.chunks import markdown_source            # noqa: E402

STEM = ROOT / "results/retrieval_accuracy/t_s3c_split_labelabl"
pages = json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text())
ta = {r["query_id"]: r for r in map(json.loads, open(f"{STEM}_type_accuracy.jsonl")) if r["query_type"] == "single_cell"}
rec = {r["query_id"]: r for r in map(json.loads, open(f"{STEM}_records.jsonl"))}
assert len(ta) == 991
miss = sorted(q for q, r in ta.items() if not r["retrieval_success"])
assert len(miss) == 85

_tab = {}


def table(tid):
    if tid not in _tab:
        _tab[tid] = hg.load_table(tid, str(ROOT / "data/hitab"))
    return _tab[tid]


def title(tid):
    return with_page_title(table(tid).title, pages.get(tid))


found, missed, cells_of = [], [], {}
for q in miss:
    t, r = ta[q], rec[q]
    (gold,) = t["gold_cell_ids"]
    got = [tuple(c) for c in t["retrieved_cell_ids"]]
    assert len(got) == 20 and tuple(gold) not in got
    assert sorted(tuple(c) for u in r["context_units"] for c in u["cells"]) == sorted(got)
    top1 = r["context_units"][0]["cells"][0]
    cells_of[q] = (tuple(gold), tuple(top1))
    in_ctx = any(c[0] == gold[0] for c in got)
    assert in_ctx == bool(r["gold_table_in_context"]) and r["table_id"] == gold[0]
    row = {"query_id": q, "question": r["question"], "gold_table_id": gold[0], "gold_table_title": title(gold[0]),
           "top1_table_id": top1[0], "top1_table_title": title(top1[0])}
    head = r["context_units"][0]["text"].startswith(f"In the table '{row['top1_table_title']}'")
    assert head if row["top1_table_title"] else not r["context_units"][0]["text"].startswith("In the table"), q
    if in_ctx:
        row["gold_rank_cut21"] = 21
        row["gold_rank_records"] = r["gold_rank"]
        found.append(row)
    else:
        row["same_title"] = "예" if row["top1_table_title"] == row["gold_table_title"] else "아니오"
        missed.append(row)


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


write("gold_table_missed.csv", missed)
write("gold_table_found.csv", found)

ranks = [x["gold_rank_records"] for x in found]
bins = [("21-30", 21, 30), ("31-50", 31, 50), ("51-100", 51, 100), ("101-200", 101, 200), ("201-500", 201, 500)]
out = {
    "records": f"{STEM}_type_accuracy.jsonl", "records_rank": f"{STEM}_records.jsonl",
    "n_single_cell": len(ta), "n_wrong": len(miss),
    "gold_table_found": len(found), "gold_table_missed": len(missed),
    "missed_top1_same_title": sum(x["same_title"] == "예" for x in missed),
    "missed_top1_same_table_id": sum(x["top1_table_id"] == x["gold_table_id"] for x in missed),
    "found_rank_cut21": dict(Counter(x["gold_rank_cut21"] for x in found)),
    "found_rank_records": {**{b: sum(lo <= k <= hi for k in ranks if k) for b, lo, hi in bins},
                           ">500": sum(k is None for k in ranks)},
    "found_rank_records_sorted": sorted(k for k in ranks if k),
}
# ---- 4) 판정용 CSV
test_tids = {r["table_id"] for r in rec.values()}
assert len(test_tids) == 538
n_title = Counter(title(t) for t in test_tids)


def path(c):
    t = table(c[0]).table
    return " > ".join([*t.row_path(c[1]), *t.col_path(c[2])])


def save_md(tid, name):
    tab = table(tid)
    head, *rows = markdown_source(tab, tab.table, title(tid))[0].split("\n")
    ncol = len(tab.raw["texts"][0])
    (TDIR / name).write_text("\n".join([head, "", rows[0], "|" + " --- |" * ncol, *rows[1:]]) + "\n", encoding="utf-8")
    return f"{TDIR.name}/{name}"


TDIR = OUT / "judge40_tables"
TDIR.mkdir(exist_ok=True)
with open(OUT / "judge40.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["번호", "query_id", "질문 원문", "정답 값", "정답 표 제목", "정답 셀 경로(행 머리글 > 열 머리글)", "정답 표 파일",
                "1위 셀의 표 제목", "1위 셀 경로(행 머리글 > 열 머리글)", "1위 표 파일",
                "정답 표 제목과 같은 제목의 test 표 수(정답 표 포함)",
                "(가) 질문만 보고 정답 표를 특정할 수 있는가(예/아니오)", "(나) 예라면 근거 단어",
                "(다) 1위 표로도 질문에 답할 수 있는가(예/아니오)"])
    for n, x in enumerate(missed, 1):
        g, t1 = cells_of[x["query_id"]]
        tt = table(t1[0]).table
        assert rec[x["query_id"]]["context_units"][0]["text"] == ra.cell_unit(
            title(t1[0]), tt.row_path(t1[1]), tt.col_path(t1[2]), tt.data[t1[1]][t1[2]], "s3c"), x["query_id"]
        stem = f"{n:02d}_{x['query_id']}"
        w.writerow([n, x["query_id"], x["question"], ", ".join(map(str, rec[x["query_id"]]["answer"])),
                    x["gold_table_title"], path(g), save_md(g[0], f"{stem}_gold.md"),
                    x["top1_table_title"], path(t1), save_md(t1[0], f"{stem}_top1.md"),
                    n_title[x["gold_table_title"]], "", "", ""])
out["judge40"] = {"rows": len(missed), "test_tables": len(test_tids),
                  "same_title_test_tables_hist": dict(sorted(Counter(n_title[x["gold_table_title"]] for x in missed).items())),
                  "gold_title_empty": sum(x["gold_table_title"] == "" for x in missed),
                  "top1_title_empty": sum(x["top1_table_title"] == "" for x in missed)}
(OUT / "analyze.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
