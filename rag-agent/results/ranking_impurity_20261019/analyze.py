"""검색 순위의 불순물 분석(882, 상위 20) — 미팅 2026-10-19 질문 "랭킹에 왜 불순물이 있냐". 정답 아닌 상위 셀이 어디에 있고(같은 행/열/표/다른 표), 왜 점수가 높은지(질문과 겹치는 머리글 낱말), 빈 값 셀 비율, 정답 셀 순위 분포.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/ranking_impurity_20261019/analyze.py
"""
import json, re, sys, collections, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "results/format_dev_20261017"))
from format_render import load_split, strip_tbl  # noqa: E402
ids = [json.loads(l)["query_id"] for l in open(ROOT / "results/cellform_20261012/A_chunk_md_records.jsonl", encoding="utf-8")]
r100 = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl", encoding="utf-8")) if r["query_id"] in set(ids) and "doc" in r}
tables, hdr, sent, coord = load_split("train", ids)
STOP = set("the of in for and to a an is are was were what which how many much does do did total amount value between from by with on at as that this it its their year years".split())
words = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3 and w not in STOP}
EMPTY = {"", "—", "–", "-", "n/a", "N/A", "nm", "NM", "*"}
val_of = lambda txt: txt.rsplit(":", 1)[-1].strip() if ":" in txt else ""
K = 20
pos = collections.Counter(); why = collections.Counter(); empty_by_rank = collections.Counter(); first_gold = []; gold_ranks = []; n_nongold = 0; same_row_cells_total = 0
for q in ids:
    R = r100[q]; S = sent[q]; C = coord[q]; rank = [u[0] for u in R["doc"]["ranked_units"]]; gold = set(R["gold_ids"]); qw = words(R["question"])
    gr = [i + 1 for i, u in enumerate(rank) if u in gold]; first_gold.append(gr[0] if gr else 999); gold_ranks += gr if gr else []
    gpos = {C[g] for g in gold if g in C}; grows = {(t, i) for t, i, j in gpos}; gcols = {(t, j) for t, i, j in gpos}; gt = {t for t, i, j in gpos}
    for r, u in enumerate(rank[:K], 1):
        if val_of(S[u]) in EMPTY: empty_by_rank["1-5" if r <= 5 else "6-20"] += 1
        if u in gold: continue
        n_nongold += 1; t, i, j = C[u]
        pos["같은 행" if (t, i) in grows else "같은 열" if (t, j) in gcols else "같은 표 다른 자리" if t in gt else "다른 표"] += 1
        tb = tables[t].table; rw = words(" ".join(map(str, strip_tbl(tb.row_path(i))))); cw = words(" ".join(map(str, strip_tbl(tb.col_path(j)))))
        a, b = bool(rw & qw), bool(cw & qw)
        why["행·열 머리글 모두 질문과 겹침" if a and b else "행 머리글만 겹침" if a else "열 머리글만 겹침" if b else "겹치는 낱말 없음"] += 1
N = len(ids)
print(f"문항 {N}, 상위 {K} 중 정답 아닌 셀 {n_nongold}({100*n_nongold/(N*K):.1f}%)")
print("위치:", {k: f"{v} ({100*v/n_nongold:.0f}%)" for k, v in pos.most_common()})
print("왜 높은가(질문과 겹치는 머리글 낱말):", {k: f"{v} ({100*v/n_nongold:.0f}%)" for k, v in why.most_common()})
print("빈 값 셀(상위 1~5 / 6~20):", dict(empty_by_rank), f"/ 882×5, 882×15")
fg = sorted(first_gold); print(f"첫 정답 셀 순위: 중앙값 {statistics.median(fg)}, 1위 {sum(r==1 for r in fg)}, ≤5 {sum(r<=5 for r in fg)}, ≤20 {sum(r<=20 for r in fg)}, 100 밖 {sum(r>100 for r in fg)} / {N}")
print(f"정답 셀 전체 {len(gold_ranks)}개의 순위: 중앙값 {statistics.median(gold_ranks)}, ≤20 {sum(r<=20 for r in gold_ranks)}, ≤100 {len(gold_ranks)}(100 안에 든 것만 셈)")
