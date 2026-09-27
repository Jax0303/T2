"""MultiHiertt train 문서 안, 정답 셀 1개 문항(조회 212 · 산술 71)의 문항별 행. 새 검색·임베딩 없음.

버전
  최종(v3.3u) = results/rerun_20260926/mh/mh_train_s3c_records.jsonl. 1위 = doc.context[0] 이 정답 칸 문장과 같음
    (rank1.py 와 같은 판정). 1위 칸 = 같은 문서에서 그 문장을 가진 색인 칸(최종 버전은 문서 안 문장이 서로 다름).
  처음(v1) = results/mh_arms/mh_train_cell_hv1_none_doc_kladder_records.jsonl 의 correct_at.doc['1'].
    처음 버전이 채점하지 못한 조회 1문항은 행을 만들지 않는다(경로가 없다).
    처음 버전 1위 칸 위치(2026-09-28 T4b 추가) = doc.context[0] 문장을 가진 같은 문서의 색인 칸(정답 칸 제외)의 위치.
    그 칸들의 위치가 갈리면 '판정불가'.
머리글 답 = rank1.py(aac902b) 의 is_header(정답, 정답 칸 행 경로 + 열 경로) 그대로.
비교 단어 분류 = CATS 를 위에서부터 보고 처음 맞는 분류. 어느 것에도 안 맞으면 '비교 없음'.
  단어 목록은 why_rank1.py 의 VC 와 합집합이 같다(2026-09-28 결과를 본 뒤 만든 목록).
단어 = rag_agent.retrieve.encoders._tokenize − sklearn ENGLISH_STOP_WORDS, 서로 다른 단어
  (results/components_20260925/hitab_path_overlap.py, results/lookup_vs_arith_20260928/analyze.py 와 같음).
경로 단어 = 행 경로 + 열 경로 원소에서 식별자('Table k', 'row N', 'column N')를 뺀 단어 (analyze.py 의 MH 경로와 같음).
m = 정답 칸과 같은 표의 색인 칸 중 경로 단어가 Q 를 모두 포함하는 칸 수(정답 칸 포함), Q = 질문 단어 ∩ 정답 칸 경로 단어.
  Q 가 비면 m = 그 표의 색인 칸 수. 색인 칸 = build_corpus(... 'cell' ...) 의 칸 = 값이 비지 않은 데이터 칸.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rank1 import ROOT, jl, is_header, gold_paths, ma, build_corpus   # noqa: E402
from rag_agent.retrieve.encoders import _tokenize                      # noqa: E402

CATS = [("두 번째", r"\bsecond\b"),
        ("기준값 비교", r"\b(exceed\w*|greater than|less than|more than|lower than|higher than|larger than|"
                     r"smaller than|in the range|between)\b"),
        ("가장 큰", r"\b(greatest|largest|highest|most|biggest|maximum|max)\b"),
        ("가장 작은", r"\b(lowest|least|smallest|minimum|min)\b")]
NONE = "비교 없음"
IDENT = re.compile(r"^(Table \d+|row \d+|column \d+)$")
SRC = {"v3.3u": "results/rerun_20260926/mh/mh_train_s3c_records.jsonl",
       "v1": "results/mh_arms/mh_train_cell_hv1_none_doc_kladder_records.jsonl"}
words = lambda s: set(_tokenize(s)) - ENGLISH_STOP_WORDS


def comparison_class(question):
    return next((name for name, pat in CATS if re.search(pat, question, re.I)), NONE)


def path_words(t, cell):
    tab = t[cell[0]].table
    return words(" ".join(x for x in [*tab.row_path(cell[1]), *tab.col_path(cell[2])] if not IDENT.match(x)))


def load(version):
    queries, docs, _ = ma.load_population("train")
    t, h = ma.build_tables(docs, version, "none")
    texts, covers, *_ = build_corpus("", sorted(t), "s3c", "cell", {}, 1000, "leaf", "values", 200, None,
                                     load=lambda tid, _d: t.get(tid))
    cells = []
    for c in covers:
        assert len(c) == 1
        cells.append(next(iter(c)))
    return t, gold_paths(queries, t, h), cells, texts


def rows(version):
    t, qs, cells, texts = load(version)
    text_of = dict(zip(cells, texts))
    by_table = defaultdict(list)
    for c in cells:
        by_table[c[0]].append(c)
    cells_of = defaultdict(list)                   # (문서, 문장) -> 칸들. 최종 버전은 1개씩, 처음 버전은 같은 문장이 여럿일 수 있다
    for c, x in zip(cells, texts):
        cells_of[(c[0].split("::")[0], x)].append(c)
    if version == "v3.3u":
        assert all(len(v) == 1 for v in cells_of.values())
    pw = {}
    out = []
    for r in jl(SRC[version]):
        if "excluded" in r or r["layer"] not in ("lookup_m1", "arith_m1"):
            continue
        (g,) = qs[r["query_id"]]["gold"]
        tab = t[g[0]].table
        rp, cp = tab.row_path(g[1]), tab.col_path(g[2])
        Q = words(r["question"]) & path_words(t, g)
        cand = [c for c in by_table[g[0]] if Q <= pw.setdefault(c, path_words(t, c))] if Q else by_table[g[0]]
        assert g in cand
        x = {"query_id": r["query_id"], "type": "산술" if r["layer"] == "arith_m1" else "조회",
             "question": r["question"], "answer": r["answer"], "row_path": rp, "col_path": cp,
             "header_answer": is_header([r["answer"]], [*rp, *cp]),
             "comparison": comparison_class(r["question"]),
             "b20": r["doc"]["correct"], "Q": sorted(Q), "m": len(cand), "table_cells": len(by_table[g[0]])}
        tops = cells_of[(g[0].split("::")[0], r["doc"]["context"][0])]
        x["rank1"] = r["correct_at"]["doc"]["1"] if version == "v1" else int(tops == [g])
        # 1위 칸 위치. 처음 버전에서 1위 문장을 가진 칸(정답 칸 제외)이 여럿이고 위치가 갈리면 '판정불가'.
        cs = set(cand)
        ws = {"가_후보_집합_안" if c in cs else "나_같은_표_집합_밖" if c[0] == g[0] else "다_다른_표"
              for c in tops if c != g}
        assert x["rank1"] or ws
        x["top1_where"] = "정답" if x["rank1"] else ws.pop() if len(ws) == 1 else "판정불가"
        out.append(x)
    return out
