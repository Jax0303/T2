#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Corpus scale: retrieve TABLES and dump them, or retrieve CELLS directly.

Everything in this repo that compared "our cell index" against "just paste the
table" handed the table arm the gold table for free. Under that gift there is
nothing for retrieval to do -- ``RESULTS.md`` K and H-2 both land there, and the
measured table sizes say why: the biggest HiTab table is 3,095 tokens and the
biggest MultiHiertt document 2,752, so the unit ALWAYS fits a modern context.
Retrieval is not needed to make a table fit. It is needed to find WHICH table.

So the honest baseline is not a dump, it is retrieve-then-dump, and the question
is where a fixed token budget is better spent:

  dump       table-level index -> take whole tables in rank order while they fit
  cell       corpus-wide cell index (S2 header-path sentences) -> take cells
  cascade    top-1 table by the table index, then cells from inside it only
  cell2dump  rank CELLS corpus-wide, then dump the whole TABLE each top cell
             belongs to -- locate with the fine-grained index, deliver with the
             unit that carries a whole operand set
  cellrow    the same trick one level down: rank CELLS, deliver the ROW each top
             cell sits in. The index unit and the context unit are separate
             choices, and every other arm here ties them together

One budget, one encoder, one population, one pool for all three. Scored
LLM-free first, because the two things that decide the answer are visible
without a reader:

  gold_table   the gold table is represented in the context at all
  osc          EVERY gold operand cell is in the context (the §1.2 metric)
  per_cell     fraction of gold operand cells in the context

``--reader`` adds answer EM on top of them. It belongs here rather than in
``baseline_comparison_llm.py`` because there the table arm is handed the gold
table: the token cost of FINDING it never appears, so "we win but spend 1.55x
the tokens" was not a fact about the method, it was the gift showing up on the
bill. Here both arms fill the same cap out of the same corpus.

Run (bm25 needs no GPU and no encoding pass):
  PYTHONPATH=. python3 scripts/corpus_dump_vs_cell.py --retriever bm25 --split dev
  PYTHONPATH=. python3 scripts/corpus_dump_vs_cell.py --retriever hybrid --split dev
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from collections import Counter, defaultdict

from rag_agent.bench import population as pop_mod
from rag_agent.bench.hitab import load_queries
from rag_agent.eval.metrics import hitab_exact_match_text
from rag_agent.generate.answerer import (
    _DIRECT_SYS, _CODEGEN_SYS, _extract_code, _safe_exec,
)
from rag_agent.llm.factory import build_llm
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax
from rag_agent.runenv import guard_resume, run_env
from rag_agent.serialization.base import Chunk

from baseline_comparison_llm import Budget, markdown_table, row_chunks
from manual_sentence_ceiling import mcnemar
from point3_reconstruction_cost import build_table_paths, cell_text
from rag_agent.serialization.caption import caption_sentence
from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,
                                              STRUCTURAL_COMPACT)

ARMS = ("dump", "cell", "cascade", "cell2dump", "cellrow", "row", "flat",
        "group", "capped", "goldcell")


class _CachedEncoder:
    """Memoize a corpus encoding on disk, keyed by the exact texts it covers.

    The key is a hash of the text list, so a corpus that changed for any reason
    -- a different split, a serialization fix, one more table -- misses the cache
    instead of silently reusing vectors that no longer describe it.
    """

    def __init__(self, inner, cache_dir: str, tag: str):
        # the tag carries a model name, and those hold slashes
        self.inner, self.dir = inner, Path(cache_dir)
        self.tag = tag.replace("/", "_")

    def encode(self, texts):
        from hashlib import md5
        # the wrapped encoder also serves the per-query encode, which is one text,
        # unique to the query and cheap: caching those writes an .npy per query
        # for a vector nothing reuses
        if len(texts) < 2:
            return self.inner.encode(texts)
        key = md5(("\x00".join(texts)).encode()).hexdigest()[:16]
        f = self.dir / f"{self.tag}_{len(texts)}_{key}.npy"
        if f.exists():
            print(f"[cache] hit {f.name}", flush=True)
            return np.load(f)
        emb = self.inner.encode(texts)
        # first write of the run creates the cache dir
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, emb)
        print(f"[cache] wrote {f.name}", flush=True)
        return emb


class _NoDense:
    """Encoder stub for the bm25 arm.

    ``HybridIndex`` falls back to ``default_encoder()`` when handed None, so a
    lexical-only run would still embed every cell in the corpus -- 58k of them
    here, with a model that is not even the one ``--embed-model`` names. The
    zero matrix keeps the index's shape contract without paying for vectors no
    scoring path reads at alpha=0.
    """

    def encode(self, texts):
        return np.zeros((len(texts), 1), dtype=np.float32)


ALPHA = {"bm25": 0.0, "dense": 1.0, "hybrid": 0.5}


def table_index_text(raw, pt, bt, mode: str = "full") -> str:
    """What a TABLE-level retriever indexes: title/caption plus the header labels.

    Deliberately not the whole table body. A table-level index that contained
    every cell would be the cell index with extra steps, and no table retriever
    in the wild embeds a 3,000-token body into one vector.

    ``mode="headers_only"`` drops the title and caption. That is not a variant
    anyone would deploy -- it is the control for reading the two datasets against
    each other, because MultiHiertt's parsed HTML carries no title and its dump
    arm is therefore already running header-only. Without this the cross-dataset
    gap cannot be told apart from the handicap.
    """
    heads = {lab for d in ("gold_rp", "gold_cp") for p in pt[d].values()
             for lab in p if lab}
    if mode == "headers_only":
        return " ".join(sorted(heads))
    title = " ".join(str(x) for x in (raw.get("title"), raw.get("caption")) if x)
    return " | ".join([title, " ".join(sorted(heads))]).strip()


@dataclass
class Corpus:
    """What both datasets have to hand main(): tables, their cells, the queries.

    Keeping this shape identical for HiTab and MultiHiertt is what lets the
    budget/table-size curve be plotted on one axis -- if the two legs differed in
    how a table or a gold cell is defined, an overlap between their curves would
    not mean anything.
    """
    tids: list                     # stable table ids
    md_lines: dict                 # tid -> markdown lines of the whole table
    table_text: dict               # tid -> what a table-level index holds
    shape: dict                    # tid -> (n_rows, n_cols) of the DATA area
    cell_text: list                # per corpus cell, its S2 sentence
    cell_owner: list               # per corpus cell, (tid, i, j)
    queries: list                  # dicts: query_id, question, gold_table, gold_cells
    # the two published-practice baselines, indexed corpus-wide like the rest so
    # they pay the same cost of FINDING the table that the oracle-gated runs
    # (RESULTS K, H-2) waived for them
    flat_text: list                # per corpus cell, its leaf-label-only sentence
    row_text: list                 # per corpus data row, its chunk
    row_owner: list                # per corpus data row, (tid, i)
    # S3 states the table's title inside every cell sentence. Only HiTab
    # ships one; MultiHiertt tables are HTML inside a document and AIT-QA
    # has no title field, so there S3 differs from S2 by phrasing alone.
    title: dict
    cell_paths: list               # per corpus cell, (row_path, col_path)



def table_labels(C) -> dict:
    """표마다 그 표의 헤더 라벨 전체(깊이 무관).

    S2h가 고르는 후보집합이다. 최상위 축 레이블로 좁히면 AIT-QA에서 무너진다 --
    그 코퍼스는 행 헤더가 빈 표가 많고 depth-0 열 라벨이 'Year' 같은 흔한 것뿐이라
    표를 특정하지 못한다 (T0 실측: 표 넘는 충돌 26.9% -> 16.4%, 관문 13.5% 미달).
    깊이를 열면 같은 k에서 8.35%까지 내려간다.
    """
    labs = defaultdict(set)
    for (rp, cp, _v), (tid, _i, _j) in zip(C.cell_paths, C.cell_owner):
        labs[tid].update(rp)
        labs[tid].update(cp)
    return labs


def label_doc_freq(C) -> Counter:
    """라벨 -> 그 라벨을 헤더에 가진 표의 수. 어느 라벨이 표를 특정하는지의 기준."""
    df = Counter()
    for labs in table_labels(C).values():
        df.update(labs)
    return df


def s2h_prefixes(C, k: int = 1) -> list:
    """셀별 S2h 접두사. 구조에서 뽑은 표 단위 구별자.

    그 표의 헤더 라벨 중 (1) 이 셀의 경로에 없고 -- 이미 문장에 있는 토큰을 반복하면
    새 정보가 0이다 -- (2) 코퍼스 문서빈도가 가장 낮은, 동률이면 가장 짧은 것 ``k``개.

    드문 것부터 고르는 이유가 이 스킴의 전부다: 표를 넘는 충돌을 줄이는 것은 그 표를
    코퍼스에서 특정하는 라벨이지 아무 라벨이나가 아니다. 제목이 하던 일을 표 자신의
    헤더에서 뽑아내는 것이고, 제목과 마찬가지로 이 셀의 경로 바깥에서 온다.

    T0 실측 (LLM 없음 · 인코더 없음, scripts/corpus_discriminability.py):

        표를 넘는 주소 충돌      HiTab      AIT-QA     토큰/셀
        S2 (기준)               11.29%     26.94%     x1.000
        S2h k=1                  2.24%      8.35%     x1.13 / x1.18   <- 배선된 값
        S2h k=2                  2.24%      8.05%     x1.36 / x1.48   (용량 관문 초과)

    k=1이 두 코퍼스에서 충돌 관문(절반 이하)과 용량 관문(+25% 이내)을 동시에 통과하는
    유일한 지점이라 고른 것이지 EM을 보고 고른 것이 아니다 -- T0은 리더 전에 닫힌다.

    주의: 문서빈도는 코퍼스 통계다. BM25의 IDF와 같은 의미에서 색인 시점 통계이며,
    표가 추가되면 접두사가 바뀌므로 재색인이 필요하다. 외부 지식은 0이다.
    """
    labs = table_labels(C)
    df = label_doc_freq(C)
    out = []
    for (rp, cp, _v), (tid, _i, _j) in zip(C.cell_paths, C.cell_owner):
        extra = labs[tid] - set(rp) - set(cp)
        pick = sorted(sorted(extra, key=lambda l: (df[l], len(l.split()), l))[:k])
        out.append(f"[{' | '.join(pick)}] " if pick else "")
    return out


def build_corpus(data_dir: str, split: str):
    """(tables, paths, raws) for every table of ``split`` whose header tree builds."""
    queries, tables = load_queries(data_dir, split)
    raw_dir = Path(data_dir) / "data/tables/raw"
    paths, raws = {}, {}
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is not None:
            paths[tid], raws[tid] = pt, raw
    return queries, tables, paths, raws


def hitab_corpus(data_dir: str, split: str, population: str,
                 max_tables: int = 0, seed: int = 42,
                 table_index: str = "full") -> Corpus:
    queries, tables, paths, raws = build_corpus(data_dir, split)
    pop = [q for q in queries if q.gold_table_id in paths and q.gold_operands]
    pop = pop_mod.pin(population, pop) if population else pop
    tids = sorted(paths)
    # Shrink the CORPUS, not the question set: cell retrieval's edge over the
    # whole-table baselines grows with the pool it searches, so HiTab's 424
    # tables against AIT-QA's 113 is a difference the datasets never agreed to
    # control. Sampling tables and keeping the queries whose gold survives holds
    # the task fixed and moves only the haystack. Applied AFTER pin, so the
    # freeze is still checked in full before this subsets it.
    if max_tables and max_tables < len(tids):
        keep = set(random.Random(seed).sample(tids, max_tables))
        pop = [q for q in pop if q.gold_table_id in keep]
        tids = [t for t in tids if t in keep]
    md, ttext, shape, ctext, owner = {}, {}, {}, [], []
    ftext, rtext, rowner, titles, cpaths = [], [], [], {}, []
    for tid in tids:
        raw, pt, bt = raws[tid], paths[tid], tables[tid]
        titles[tid] = " ".join(str(x) for x in
                               (raw.get("title"), raw.get("caption")) if x)
        md[tid] = markdown_table(raw, max(1, len(raw["texts"]) - pt["n_r"]))
        ttext[tid] = table_index_text(raw, pt, bt, table_index)
        shape[tid] = (pt["n_r"], pt["n_c"])
        for i in range(pt["n_r"]):
            for j in range(pt["n_c"]):
                ctext.append(cell_text(pt["gold_rp"][i], pt["gold_cp"][j],
                                       bt.data[i][j], "S2"))
                ftext.append(cell_text(pt["gold_rp"][i], pt["gold_cp"][j],
                                       bt.data[i][j], "flat"))
                cpaths.append((list(pt["gold_rp"][i]), list(pt["gold_cp"][j]),
                               bt.data[i][j]))
                owner.append((tid, i, j))
        for i, txt in enumerate(row_chunks(bt, pt)):
            rtext.append(txt)
            rowner.append((tid, i))
    qs = [{"query_id": q.query_id, "question": q.question, "answer": q.answer,
           "gold_table": q.gold_table_id,
           "gold_cells": {(q.gold_table_id, o.row, o.col) for o in q.gold_operands}}
          for q in pop]
    return Corpus(tids, md, ttext, shape, ctext, owner, qs, ftext, rtext,
                  rowner, titles, cpaths)


def aitqa_corpus(data_dir: str = "data/aitqa",
                 population: str = "aitqa_answer_matched") -> Corpus:
    """AIT-QA (Katsis et al., NAACL 2022 industry): airline tables, 113 of them.

    The third hierarchical benchmark, and the one that needs no reconstruction:
    it ships ``column_header``/``row_header`` as ancestor LISTS, so the flat-vs-path
    contrast is read straight off the annotation. Gold cells are not annotated, so
    they are recovered by matching the answer strings against cell values, and any
    question whose match count differs from its answer count is dropped rather than
    scored against an over-inclusive gold set.
    """
    import re

    def norm(s):
        return re.sub(r"[\s,$%]", "", str(s)).strip().lower()

    tabs = {d["id"]: d for d in
            (json.loads(l) for l in open(f"{data_dir}/aitqa_tables.jsonl"))}
    qs_raw = [json.loads(l) for l in open(f"{data_dir}/aitqa_questions.jsonl")]

    tids = sorted(tabs)
    md, ttext, shape, ctext, owner = {}, {}, {}, [], []
    ftext, rtext, rowner, cpaths = [], [], [], []
    for tid in tids:
        t = tabs[tid]
        ch, rh, data = t["column_header"], t["row_header"], t["data"]
        n_r, n_c = len(data), max((len(r) for r in data), default=0)

        def cp_of(j):
            return [str(x).strip() for x in (ch[j] if j < len(ch) else []) if str(x).strip()]

        def rp_of(i):
            return [str(x).strip() for x in (rh[i] if i < len(rh) else []) if str(x).strip()]

        head = [" / ".join(cp_of(j)) for j in range(n_c)]
        md[tid] = (["| " + " | ".join(head) + " |", "|" + "---|" * n_c]
                   + ["| " + " | ".join(str(x) for x in r) + " |" for r in data])
        ttext[tid] = " | ".join(sorted({lab for j in range(n_c) for lab in cp_of(j)}
                                       | {lab for i in range(n_r) for lab in rp_of(i)}))
        shape[tid] = (n_r, n_c)
        for i, row in enumerate(data):
            rp, cells = rp_of(i), []
            for j in range(n_c):
                v = row[j] if j < len(row) else ""
                ctext.append(cell_text(list(rp), cp_of(j), v, "S2"))
                ftext.append(cell_text(list(rp), cp_of(j), v, "flat"))
                cpaths.append((list(rp), cp_of(j), v))
                owner.append((tid, i, j))
                cells.append(f"{(cp_of(j) or [''])[-1]}: {v}")
            joined = " | ".join(cells)
            rtext.append(f"{' > '.join(rp)} | {joined}" if rp else joined)
            rowner.append((tid, i))

    qs = []
    for q in qs_raw:
        t = tabs.get(q["table_id"])
        if t is None:
            continue
        want = [norm(a) for a in q["answers"] if norm(a)]
        found = {(q["table_id"], i, j) for i, row in enumerate(t["data"])
                 for j, v in enumerate(row) if norm(v) in set(want)}
        if not found or len(found) != len(set(want)):
            continue                      # unresolved, or the match is ambiguous
        qs.append({"query_id": q["id"], "question": q["question"],
                   "answer": q["answers"][0], "gold_table": q["table_id"],
                   "gold_cells": found})
    # 답 문자열 매칭은 정규화 규칙에 의존하므로 모집단이 코드와 함께 움직인다.
    # 고정된 목록에 pin 해서, 정규화가 바뀌면 수치가 아니라 실행이 실패하게 한다.
    if population:
        qs = pop_mod.pin(population, qs)
    return Corpus(tids, md, ttext, shape, ctext, owner, qs, ftext, rtext,
                  rowner, {t: '' for t in tids}, cpaths)


def _rhb_split_title(grid):
    """Peel the free-text title rows off the top of a spreadsheet grid.

    Statistical spreadsheets put the table's name in its own full-width row
    ("Table A-6. Time spent in primary activities ..."), which an Excel -> HTML
    export turns into a row whose only filled cell is column 0. RealHiTBench
    ships tables that way, so the title has to be RECOVERED from the grid rather
    than read off a field -- and it is present for only about half the corpus,
    which is exactly what makes this dataset the within-corpus title contrast
    that HiTab (99.3% titled) vs MultiHiertt/AIT-QA (0%) cannot be.

    Stripping matters for correctness, not tidiness: left in place, the title row
    is counted by ``guess_n_header_rows`` as a header level and
    ``_hierarchical_carry`` prepends it to EVERY column path. S2 would then carry
    the title for free and the S2-vs-S3 contrast would measure nothing.

    Returns ``(title, grid_without_title)``.
    """
    title, n_strip = "", 0
    for row in grid[:6]:
        filled = [str(c).strip() for c in row if str(c).strip()]
        if len(filled) > 1:
            break                     # two labels on one row: the header band
        n_strip += 1                  # 0 or 1 filled cell: preamble, not data
        t = filled[0] if filled else ""
        # The FIRST prose line above the table is its name. The ones after it are
        # boilerplate that repeats across the corpus ("Back to contents", "This
        # worksheet contains one table", "Source: ..."), and taking those as part
        # of the title would hand every cell of the table the same junk string.
        # A bare number or a short code is not a name either.
        if not title and len(t) >= 15 and " " in t:
            title = t
    # a grid that is ALL preamble is not a table; keep at least three rows
    if n_strip and len(grid) - n_strip >= 3:
        return title, grid[n_strip:]
    return "", grid


def realhitbench_corpus(data_dir: str = "data/realhitbench",
                        subqtypes: tuple = ()) -> Corpus:
    """RealHiTBench (Zhang et al., ACL Findings 2025; arXiv:2506.13405).

    The fourth hierarchical benchmark and the first where the TITLE varies inside
    one corpus -- see :func:`_rhb_split_title`. Tables are Excel -> HTML exports
    with ``rowspan``/``colspan`` intact, so the same reconstruction front-end the
    other corpora use applies unchanged.

    Gold cells are not annotated. As in :func:`aitqa_corpus` they are recovered by
    matching the answer strings against cell values, and a question whose match
    count differs from its answer count is dropped rather than scored against an
    over-inclusive gold set. ``subqtypes`` defaults to every type, because the
    answer match is what defines the population: whatever the dataset called a
    question, it survives only if its answer resolves to a unique set of cells.

    Cells with an EMPTY value are not indexed. An Excel -> HTML export ships the
    sheet's whole used range, so 55.9% of the grid cells here hold nothing, while
    the other three corpora are dense and have no such cells. Indexing a blank is
    indexing noise, and dropping it costs every arm the same.
    """
    import re

    from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                       parse_html_table_with_merges,
                                       reconstruct_col_paths, reconstruct_row_paths)

    def norm(s):
        return re.sub(r"[\s,$%]", "", str(s)).strip().lower()

    qa = json.load(open(f"{data_dir}/QA_final.json"))["queries"]
    want = set(subqtypes)
    qs_raw = [q for q in qa if not want or q.get("SubQType") in want]

    md, ttext, shape, ctext, owner = {}, {}, {}, [], []
    ftext, rtext, rowner, titles, cpaths = [], [], [], {}, []
    data_of, tids = {}, []
    for tid in sorted({q["FileName"] for q in qs_raw}):
        f = Path(data_dir) / "html" / f"{tid}.html"
        if not f.exists():
            continue
        try:
            grid, _ = parse_html_table_with_merges(f.read_text(errors="replace"))
        except Exception:
            continue
        if not grid or len(grid) < 3 or len(grid[0]) < 2:
            continue
        title, grid = _rhb_split_title(grid)
        nhc = guess_n_header_cols(grid)
        nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=nhc), len(grid) - 1))
        cols = reconstruct_col_paths(grid, nhr, n_header_cols=nhc)
        rows = reconstruct_row_paths(grid, nhr, n_header_cols=nhc)
        data = [[str(x).strip() for x in r[nhc:]] for r in grid[nhr:]]
        if not data or not data[0]:
            continue
        tids.append(tid)
        titles[tid] = title
        data_of[tid] = data
        # the title row is part of the table as shipped, so the whole-table arm
        # keeps it -- the cell arms have to earn the title by repeating it
        md[tid] = ([f"| {title} |"] if title else []) + \
                  ["| " + " | ".join(str(x) for x in r) + " |" for r in grid[:nhr]] + \
                  ["|" + "---|" * len(grid[0])] + \
                  ["| " + " | ".join(str(x) for x in r) + " |" for r in grid[nhr:]]
        heads = {lab for p in list(rows) + list(cols) for lab in p if lab}
        ttext[tid] = " ".join(x for x in (title, " | ".join(sorted(heads))) if x)
        shape[tid] = (len(data), len(data[0]))
        for i, row in enumerate(data):
            rp = list(rows[i]) if i < len(rows) else []
            cells = []
            for j, v in enumerate(row):
                cp = list(cols[j]) if j < len(cols) else []
                cells.append(f"{(cp or [''])[-1]}: {v}")
                if not str(v).strip():
                    continue          # blank sheet padding, see the docstring
                ctext.append(cell_text(list(rp), cp, v, "S2"))
                ftext.append(cell_text(list(rp), cp, v, "flat"))
                cpaths.append((list(rp), cp, v))
                owner.append((tid, i, j))
            joined = " | ".join(cells)
            rtext.append(f"{' > '.join(rp)} | {joined}" if rp else joined)
            rowner.append((tid, i))

    qs = []
    for q in qs_raw:
        tid = q["FileName"]
        if tid not in data_of:
            continue
        ans = str(q.get("ProcessedAnswer") or q.get("FinalAnswer") or "")
        want_v = [norm(a) for a in ans.split(",") if norm(a)]
        if not want_v:
            continue
        found = {(tid, i, j) for i, row in enumerate(data_of[tid])
                 for j, v in enumerate(row) if norm(v) in set(want_v)}
        if not found or len(found) != len(set(want_v)):
            continue                      # unresolved, or the match is ambiguous
        qs.append({"query_id": str(q["id"]), "question": q["Question"],
                   "answer": ans, "gold_table": tid, "gold_cells": found})
    return Corpus(tids, md, ttext, shape, ctext, owner, qs, ftext, rtext,
                  rowner, titles, cpaths)


def multihiertt_corpus(n_queries: int, seed: int) -> Corpus:
    """Same shape from MultiHiertt, where a table is an HTML string in a document.

    MultiHiertt carries no global table id -- each row ships its own document's
    tables -- so identity is the hash of the HTML. Documents repeat across
    questions, and deduplicating on that hash is what turns the sample into one
    corpus rather than one pool per question.
    """
    from hashlib import md5

    from baseline_comparison_multihiertt import load_population, parse_doc

    pop, _ = load_population(n_queries, seed)
    md, ttext, shape, ctext, owner, qs = {}, {}, {}, [], [], []
    ftext, rtext, rowner, cpaths = [], [], [], []
    seen = set()
    for q in pop:
        tabs = parse_doc(q)
        if tabs is None:
            continue                      # gold cell unparseable for every arm
        local = {}
        for t_idx, t in tabs.items():
            tid = md5(q["tables"][t_idx].encode()).hexdigest()[:16]
            local[t_idx] = tid
            if tid in seen:
                continue
            seen.add(tid)
            grid, nhr, nhc = t["grid"], t["nhr"], t["nhc"]
            w = max(len(r) for r in grid)
            g = [[(x or "").strip() for x in r] + [""] * (w - len(r)) for r in grid]
            md[tid] = (["| " + " | ".join(r) + " |" for r in g[:nhr]]
                       + ["|" + "---|" * w]
                       + ["| " + " | ".join(r) + " |" for r in g[nhr:]])
            heads = {lab for p in list(t["rows"]) + list(t["cols"]) for lab in p if lab}
            ttext[tid] = " ".join(sorted(heads))
            shape[tid] = (len(g) - nhr, w - nhc)
            for r in range(nhr, len(g)):
                rp = t["rows"][r - nhr] if (r - nhr) < len(t["rows"]) else []
                cells = []
                for c in range(nhc, w):
                    cp = t["cols"][c - nhc] if (c - nhc) < len(t["cols"]) else []
                    ctext.append(cell_text(list(rp), list(cp), g[r][c], "S2"))
                    ftext.append(cell_text(list(rp), list(cp), g[r][c], "flat"))
                    cpaths.append((list(rp), list(cp), g[r][c]))
                    owner.append((tid, r - nhr, c - nhc))
                    cells.append(f"{cp[-1] if cp else ''}: {g[r][c]}")
                # same shape as baseline_comparison_llm.row_chunks, built from
                # the parsed grid because MultiHiertt has no BenchTable
                joined = " | ".join(cells)
                rtext.append(f"{' > '.join(rp)} | {joined}" if rp else joined)
                rowner.append((tid, r - nhr))
        gold = {(local[t_idx], r - tabs[t_idx]["nhr"], c - tabs[t_idx]["nhc"])
                for t_idx, r, c in q["cells"] if t_idx in local}
        qs.append({"query_id": q["uid"], "question": q["question"],
                   "answer": q["answer"], "gold_table": next(iter(gold))[0], "gold_cells": gold})
    return Corpus(sorted(md), md, ttext, shape, ctext, owner, qs,
                  ftext, rtext, rowner, {t: '' for t in md}, cpaths)


def positions(order) -> np.ndarray:
    """Inverse permutation: ``pos[i]`` is where unit ``i`` sits in ``order``.

    The cascade arm needs the retriever's rank for each cell of one table.
    Asking ``order.index(cell)`` costs a linear scan of the whole corpus per
    cell -- 58k x 120 x 175 -- so the ranking is inverted once instead.
    """
    pos = np.empty(len(order), dtype=np.int64)
    pos[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
    return pos


def rank_of(index: HybridIndex, question: str, alpha: float) -> list[int]:
    bm = index._bm25_scores(question)
    dn = index._dense_scores(question) if alpha > 0 else np.zeros_like(bm)
    if alpha == 0.0:
        return list(np.argsort(-bm))
    if alpha == 1.0:
        return list(np.argsort(-dn))
    return list(np.argsort(-(alpha * _minmax(dn) + (1 - alpha) * _minmax(bm))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "multihiertt", "aitqa", "realhitbench"])
    ap.add_argument("--mh-queries", type=int, default=400,
                    help="MultiHiertt questions to sample (their documents become "
                         "the corpus)")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--rhb-subqtypes", nargs="*", default=[],
                    help="RealHiTBench SubQType values to keep; the default keeps "
                         "every type. The population is defined by the ANSWER "
                         "MATCH, not by the label: a question survives only if its "
                         "answer strings resolve to a unique set of data cells, "
                         "which is a cell lookup whatever the dataset called it. "
                         "Filtering to Value-Matching first would cost 52%% of the "
                         "population (243 -> 116) and leave the title split at "
                         "n=45, too thin to test")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_arith")
    ap.add_argument("--budget", type=int, default=1024, help="context tokens per arm")
    ap.add_argument("--retriever", default="bm25", choices=list(ALPHA))
    ap.add_argument("--alpha", type=float, default=None,
                    help="override the retriever's dense/BM25 mix. ALPHA "
                         "names three points (0 / .5 / 1) and the measured "
                         "optimum for cell retrieval is .6-.7 "
                         "(results/alpha_sweep_prefix.json), so the value "
                         "a run used is recorded rather than implied by "
                         "the retriever name.")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--max-tables", type=int, default=0,
                    help="HiTab only: sample this many tables (seeded) and keep "
                         "only the queries whose gold table survived. Varies "
                         "corpus scale with the task held fixed -- the control "
                         "HiTab-vs-AIT-QA never had, since 424 tables against "
                         "113 confounds every cross-dataset gap. 0 = whole corpus")
    ap.add_argument("--table-index", default="full",
                    choices=["full", "headers_only"],
                    help="what the table-level index holds. headers_only is the "
                         "control that matches MultiHiertt, whose tables have no title")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell",
                    help="where corpus embeddings are memoized across budgets")
    ap.add_argument("--reader", default="",
                    help="answer the questions too, with this model (e.g. "
                         "local:Qwen/Qwen2.5-7B-Instruct). Every arm gets the same "
                         "budget and the same prompt frame, so an answer-EM "
                         "difference is a difference in what the budget bought")
    ap.add_argument("--answer-mode", default="direct", choices=["direct", "codegen"],
                    help="direct: the reader writes the final answer. codegen: the "
                         "reader writes one Python line `answer = <expr over cell "
                         "numbers>` and PYTHON computes it — a reading-side lever "
                         "for the arithmetic population, where a local 7B can pick "
                         "the cells but cannot do the mental math. Keep separate "
                         "--out/--records per mode: the two are not comparable")
    ap.add_argument("--codegen-max-tokens", type=int, default=512,
                    help="completion cap for the codegen line")
    ap.add_argument("--cell-scheme", default="S2",
                    choices=["S2", "S2r", "S2t", "S2h", "S3", "S3c", "mt2net"],
                    help="what the CELL arm indexes. S2 is the bare header path "
                         "('a > b > c: v'); S3 is this work's deployed index unit "
                         "-- a sentence stating the table title and both paths "
                         "(rag_agent/serialization/caption.py); S3c is S3 where a "
                         "title exists and S2 where none does, since the frame is "
                         "the title's seat and costs budget without one. S2t is S2 "
                         "plus a per-table tag ('t47 | a > b: v') -- address "
                         "uniqueness carrying no meaning a query could match, so "
                         "it prices uniqueness apart from topical matching. S2r is S2 "
                         "with each axis reversed to leaf-first -- the SAME tokens "
                         "in a different order, which is the only way to price "
                         "MT2Net's leaf-first reading apart from its sentence "
                         "frame. Only the cell and cascade arms change; "
                         "dump/row/flat do not read it")
    ap.add_argument("--cap", type=int, default=3,
                    help="`capped` arm: at most this many cells from any one "
                         "table. 18 of the 24 oracle-condition failures answered "
                         "from a SIBLING of the gold cell, so this trades depth "
                         "inside a table for fewer confusable neighbours")
    ap.add_argument("--no-title", action="store_true",
                    help="S3 only: render the sentence WITHOUT the table title. "
                         "S3 changes two things at once against S2 -- the title "
                         "and the natural-language frame -- and this splits them")
    ap.add_argument("--arms", default="",
                    help="comma-separated subset of arms to run, e.g. 'flat,cell'. "
                         "The arms that ignore --cell-scheme need not be paid for "
                         "twice. Default: all")
    ap.add_argument("--force-resume", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    arms = tuple(a.strip() for a in args.arms.split(',') if a.strip()) or ARMS
    bad = set(arms) - set(ARMS)
    if bad:
        ap.error(f'unknown arms {sorted(bad)}; pick from {ARMS}')
    env = run_env(args.seed, args.embed_model)
    out_path = args.out or (f"results/corpus_dump_vs_cell_{args.dataset}_{args.retriever}_{args.budget}"
      + ("" if args.table_index == "full" else "_headonly") + ".json")

    if args.dataset == "hitab":
        C = hitab_corpus(args.data_dir, args.split, args.population,
                         max_tables=args.max_tables, seed=args.seed,
                         table_index=args.table_index)
    elif args.dataset == "aitqa":
        C = aitqa_corpus()
    elif args.dataset == "realhitbench":
        C = realhitbench_corpus(subqtypes=tuple(args.rhb_subqtypes))
    else:
        C = multihiertt_corpus(args.mh_queries, args.seed)
    pop = C.queries[: args.max_queries] if args.max_queries else C.queries
    tids = C.tids
    print(f"[corpus] {len(tids)} tables / {len(C.cell_text)} cells | "
          f"[pop] {len(pop)} queries | budget {args.budget} | {args.retriever}",
          flush=True)

    llm = build_llm(args.reader) if args.reader else None

    bud = Budget(args.embed_model)
    enc = (default_encoder(model_name=args.embed_model)
           if ALPHA[args.retriever] > 0 else _NoDense())

    tbl_chunks = [Chunk(table_id=tid, chunk_id=f"t::{tid}", text=C.table_text[tid],
                        scheme="table", kind="table") for tid in tids]
    if args.cell_scheme == "S2r":
        # Word order, and nothing else. MT2Net beat S2 on HiTab's untitled
        # condition (.540 vs .430, 15:4, p=.0192) and tied it on AIT-QA, and the
        # two templates differ in BOTH word order and sentence frame. S3c already
        # removed the frame; reversing each axis to leaf-first removes the order,
        # leaving a string with the same token multiset as S2 -- so a difference
        # here cannot be capacity, the way every other template contrast can.
        C.cell_text[:] = [cell_text(rp[::-1], cp[::-1], v, "S2")
                          for rp, cp, v in C.cell_paths]
    if args.cell_scheme == "S2t":
        # Uniqueness with no meaning. Every measured gain in this repo tracks a
        # drop in ADDRESS COLLISION -- the share of cells whose sentence, minus
        # its value, is not unique -- and `stuck` (cells that still collide after
        # the path is added) gains exactly 0 on three datasets. That reads as
        # "the index unit must be uniquely resolvable". A per-table tag tests it
        # directly: it drives HiTab's S2 collision 13.12% -> 2.11% and the
        # cross-table share to 0.00%, i.e. FURTHER than the title does (9.14% /
        # 7.30%), while carrying no information a query could ever match. If
        # uniqueness is the mechanism the tag beats the title; if it buys
        # nothing, uniqueness alone is not the mechanism and the address has to
        # be query-matchable as well. The tag is an ordinal over the corpus's own
        # table order, so it is stable across runs and independent of any label.
        tag = {t: f"t{k}" for k, t in enumerate(C.tids)}
        C.cell_text[:] = [f"{tag[t]} | {cell_text(rp, cp, v, 'S2')}"
                          for (rp, cp, v), (t, i, j)
                          in zip(C.cell_paths, C.cell_owner)]
    if args.cell_scheme == "S2h":
        # 구조에서 뽑은 표 단위 구별자. S2t(무의미한 서수 태그)는 표를 넘는 충돌을
        # 0.00%까지 없애고도 AIT-QA에서 -.032로 졌다 -- 질의가 `t47`을 칠 수 없어
        # 검색을 좁히지 못했기 때문이다. S2h가 붙이는 것은 실제 헤더 단어라 좁힐
        # 자격이 있고, 그 하나가 이 스킴과 S2t의 유일한 차이다.
        C.cell_text[:] = [pre + txt for pre, txt
                          in zip(s2h_prefixes(C), C.cell_text)]
    if args.cell_scheme in ("S3", "S3c", "mt2net"):
        # re-render from the same paths the S2 text was built from, so the
        # schemes differ in rendering only and the cell SET stays identical.
        # mt2net renders the SAME cells and the SAME two paths under Zhao et al.
        # (2022) §4's sentence form -- the closest published index unit to this
        # work's, so putting it here is what lets the two be scored in one table
        # off one corpus, one budget, one reader and one scorer. That template
        # never reads the title, so --no-title does not apply to it.
        tmpl = {"mt2net": MT2NET, "S3c": STRUCTURAL_COMPACT}.get(
            args.cell_scheme, STRUCTURAL)
        C.cell_text[:] = [
            caption_sentence("" if args.no_title else C.title.get(t, ""),
                             C.cell_paths[n][0],
                             C.cell_paths[n][1], C.cell_paths[n][2],
                             template=tmpl)
            for n, (t, i, j) in enumerate(C.cell_owner)]
    cell_chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=txt,
                         scheme=args.cell_scheme, kind="cell")
                   for txt, (t, i, j) in zip(C.cell_text, C.cell_owner)]
    flat_chunks = [Chunk(table_id=t, chunk_id=f"f::{t}::{i}:{j}", text=txt,
                         scheme="flat", kind="cell")
                   for txt, (t, i, j) in zip(C.flat_text, C.cell_owner)]
    row_chunk_list = [Chunk(table_id=t, chunk_id=f"r::{t}::{i}", text=txt,
                            scheme="row", kind="row")
                      for txt, (t, i) in zip(C.row_text, C.row_owner)]
    cell_owner = C.cell_owner

    if ALPHA[args.retriever] > 0:
        # the budget sweep runs the same corpus at six budgets; without this the
        # 58k-cell encoding is paid six times for vectors that cannot differ
        enc = _CachedEncoder(enc, args.cache_dir,
                             f"{args.dataset}_{args.split}_{args.embed_model}")

    t0 = time.time()
    tbl_ix = HybridIndex(tbl_chunks, encoder=enc, alpha=0.5)
    cell_ix = HybridIndex(cell_chunks, encoder=enc, alpha=0.5)
    flat_ix = HybridIndex(flat_chunks, encoder=enc, alpha=0.5)
    row_ix = HybridIndex(row_chunk_list, encoder=enc, alpha=0.5)
    print(f"[index] built in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    cell_tok = np.array([bud.count(c.text) for c in cell_chunks], dtype=np.int32)
    flat_tok = np.array([bud.count(c.text) for c in flat_chunks], dtype=np.int32)
    row_tok = np.array([bud.count(c.text) for c in row_chunk_list], dtype=np.int32)
    print(f"[tokens] {len(cell_tok)} cells / {len(row_tok)} rows counted in "
          f"{time.time() - t0:.0f}s", flush=True)

    # (table, row) -> row-chunk index, so the cellrow arm can go from a ranked
    # cell to the row that carries it without rescanning row_owner per query
    row_of = {ro: k for k, ro in enumerate(C.row_owner)}

    # what the grouped arm renders: one header per table, one short line per cell
    grp_head = {t: f"[table] {C.title.get(t, '')}".rstrip() for t in tids}
    grp_line = [f"  {' > '.join(rp)} | {' > '.join(cp)} : {v}"
                for rp, cp, v in C.cell_paths]
    grp_head_tok = {t: bud.count(x) + 1 for t, x in grp_head.items()}
    grp_line_tok = np.array([bud.count(x) for x in grp_line], dtype=np.int32)

    cells_by_table: dict[str, list[int]] = {}
    pos_of_cell: dict[tuple, int] = {}
    for n, (tid, i_, j_) in enumerate(cell_owner):
        cells_by_table.setdefault(tid, []).append(n)
        pos_of_cell[(tid, i_, j_)] = n

    alpha = ALPHA[args.retriever] if args.alpha is None else args.alpha
    tok_cache: dict[str, int] = {}

    def md_tokens(tid: str) -> int:
        if tid not in tok_cache:
            tok_cache[tid] = bud.count("\n".join(C.md_lines[tid]))
        return tok_cache[tid]

    rec_path = Path(str(Path(out_path).with_suffix("")) + "_records.jsonl")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    guard_resume(rec_path, env,
                 reader=(f"{llm.name}#{args.answer_mode}" if llm else None),
                 population=(args.population if args.dataset == "hitab"
                             else "aitqa_answer_matched" if args.dataset == "aitqa"
                             else f"rhb_{'_'.join(args.rhb_subqtypes)}"
                             if args.dataset == "realhitbench"
                             else f"multihiertt_{args.mh_queries}_{args.seed}"),
                 cell_scheme=args.cell_scheme, cell_title=not args.no_title,
                 budget=args.budget, retriever=args.retriever, alpha=alpha,
                 max_tables=args.max_tables or None,
                 force=args.force_resume)
    # Pick up where a killed run stopped. guard_resume has already refused the
    # case where the configuration changed underneath, so whatever is on disk
    # belongs to this run; a hosted reader on a token-per-minute limit makes
    # these runs hours long, and starting from zero after every hiccup is how
    # they never finish. Only complete lines count -- a half-written last line
    # from a SIGKILL is dropped.
    recs = []
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            if not all(a in r for a in arms):
                break        # written under a different --arms; redo from here
            recs.append(r)
    done = {r["query_id"] for r in recs}
    if done:
        print(f"[resume] {len(done)} queries already on disk", flush=True)
    rec_fh = open(rec_path, "w")
    for r in recs:
        rec_fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    rec_fh.flush()

    for n_done, q in enumerate(pop, 1):
        if q["query_id"] in done:
            continue
        gold_t, gold_cells = q["gold_table"], q["gold_cells"]
        rec = {"query_id": q["query_id"], "m": len(gold_cells), "gold_table": gold_t,
               "gold_table_tokens": md_tokens(gold_t)}

        t_order = rank_of(tbl_ix, q["question"], alpha)
        c_order = rank_of(cell_ix, q["question"], alpha)
        f_order = rank_of(flat_ix, q["question"], alpha)
        r_order = rank_of(row_ix, q["question"], alpha)
        c_pos = positions(c_order)
        # the question underneath every arm: can the corpus find the gold TABLE,
        # and does a fine-grained index find it better than a table-level one?
        rec["table_rank"] = int(t_order.index(tids.index(gold_t))) + 1
        voted, seen_v = [], set()
        for p in c_order[:5000]:
            tv = cell_owner[p][0]
            if tv not in seen_v:
                seen_v.add(tv)
                voted.append(tv)
        rec["table_rank_cellvote"] = (voted.index(gold_t) + 1
                                      if gold_t in voted else 10 ** 6)

        for arm in arms:
            used, in_ctx, seen_tables, parts = 0, set(), [], []
            if arm in ("dump", "cell2dump"):
                if arm == "dump":
                    t_rank = [tids[p] for p in t_order]
                else:
                    # the cell ranking votes on tables: a table's rank is that of
                    # its best cell, so the fine index chooses and the table unit
                    # delivers
                    t_rank, seen = [], set()
                    for p in c_order[:2000]:
                        tid_ = cell_owner[p][0]
                        if tid_ not in seen:
                            seen.add(tid_)
                            t_rank.append(tid_)
                for tid in t_rank:
                    n = md_tokens(tid)
                    if used + n > args.budget:
                        continue          # same greedy rule as Budget.fill
                    used += n
                    seen_tables.append(tid)
                    parts.append("\n".join(C.md_lines[tid]))
                    n_r_, n_c_ = C.shape[tid]
                    in_ctx |= {(tid, i, j) for i in range(n_r_) for j in range(n_c_)}
            elif arm == "capped":
                per_tab: dict[str, int] = {}
                for pos in c_order:
                    tid = cell_owner[pos][0]
                    if per_tab.get(tid, 0) >= args.cap:
                        continue
                    n = int(cell_tok[pos])
                    if used + n > args.budget:
                        if used + int(cell_tok.min()) > args.budget:
                            break
                        continue
                    used += n
                    per_tab[tid] = per_tab.get(tid, 0) + 1
                    parts.append(C.cell_text[pos])
                    in_ctx.add(cell_owner[pos])
                    if tid not in seen_tables:
                        seen_tables.append(tid)
            elif arm == "group":
                # a cell costs its own line, plus the table header only if this
                # is the first cell of that table -- the saving IS the claim, so
                # the budget must be charged the way the context is rendered
                bucket: dict[str, list[int]] = {}
                for pos in c_order:
                    tid = cell_owner[pos][0]
                    n = int(grp_line_tok[pos]) + (0 if tid in bucket
                                                  else grp_head_tok[tid])
                    if used + n > args.budget:
                        if used + int(grp_line_tok.min()) > args.budget:
                            break
                        continue
                    used += n
                    if tid not in bucket:
                        bucket[tid] = []
                        seen_tables.append(tid)
                    bucket[tid].append(pos)
                    in_ctx.add(cell_owner[pos])
                for tid in seen_tables:
                    parts.append(grp_head[tid])
                    parts.extend(grp_line[p_] for p_ in bucket[tid])
            elif arm == "cellrow":
                # the cell index chooses, the row unit delivers. `row` ranks
                # rows by the row's own text, which reads as a bag of numbers;
                # this keeps the cell sentence as the thing being scored and
                # only changes what the reader is handed. Separates the INDEX
                # unit from the CONTEXT unit, which every other arm conflates.
                taken = set()
                for pos in c_order:
                    tid, i, _ = cell_owner[pos]
                    k = row_of.get((tid, i))
                    if k is None or k in taken:
                        continue
                    n = int(row_tok[k])
                    if used + n > args.budget:
                        if used + int(row_tok.min()) > args.budget:
                            break
                        continue
                    used += n
                    taken.add(k)
                    parts.append(C.row_text[k])
                    in_ctx |= {(tid, i, j) for j in range(C.shape[tid][1])}
                    if tid not in seen_tables:
                        seen_tables.append(tid)
            elif arm == "row":
                # a row chunk delivers the whole data row, so every cell of it
                # counts as retrieved -- that is the point of the unit
                for pos in r_order:
                    n = int(row_tok[pos])
                    if used + n > args.budget:
                        if used + int(row_tok.min()) > args.budget:
                            break
                        continue
                    used += n
                    parts.append(C.row_text[pos])
                    tid, i = C.row_owner[pos]
                    in_ctx |= {(tid, i, j) for j in range(C.shape[tid][1])}
                    if tid not in seen_tables:
                        seen_tables.append(tid)
            elif arm == "goldcell":
                # A CEILING, not a method: the context is exactly the gold cells'
                # own sentences and nothing else. Retrieval is perfect and there
                # are no distractors, so whatever this misses is the reader
                # failing on text it was handed. That is the number which says
                # whether ANY indexing or context-construction change still has
                # headroom -- if the ceiling sits at the OSC=1 conditional EM the
                # real runs already reach, distractors cost nothing and the wall
                # is the reader. Deliberately ignores --budget; the lookup
                # population has one gold cell, tens of tokens.
                for key in sorted(gold_cells):
                    pos = pos_of_cell.get(key)
                    if pos is None:
                        continue
                    used += int(cell_tok[pos])
                    parts.append(C.cell_text[pos])
                    in_ctx.add(key)
                    if key[0] not in seen_tables:
                        seen_tables.append(key[0])
            else:
                order, tok, text = ((f_order, flat_tok, C.flat_text) if arm == "flat"
                                    else (c_order, cell_tok, C.cell_text))
                pool = (order if arm in ("cell", "flat")
                        else sorted(cells_by_table.get(tids[t_order[0]], []),
                                    key=lambda x: c_pos[x]))
                for pos in pool:
                    n = int(tok[pos])
                    if used + n > args.budget:
                        if used + int(tok.min()) > args.budget:
                            break     # nothing left in the corpus can fit
                        continue
                    used += n
                    parts.append(text[pos])
                    tid, i, j = cell_owner[pos]
                    in_ctx.add((tid, i, j))
                    if tid not in seen_tables:
                        seen_tables.append(tid)
            hit = gold_cells & in_ctx
            n_r, n_c = C.shape[gold_t]
            gold_all = {(gold_t, i, j) for i in range(n_r) for j in range(n_c)}
            rec[arm] = {"osc": int(len(hit) == len(gold_cells)),
                        "per_cell": len(hit) / len(gold_cells),
                        # two different things the old single field conflated:
                        # a cell arm "has the gold table" with one cell of it
                        "gold_table_any": int(gold_t in seen_tables),
                        "gold_table_whole": int(gold_all <= in_ctx),
                        "tokens": used, "n_tables": len(seen_tables)}
            if llm is not None:
                if args.answer_mode == "codegen":
                    # The reader only names the cells and writes the arithmetic;
                    # Python evaluates it. Offloads the mental math a local 7B
                    # fails (osc=1 yet wrong answer) to an exact evaluator.
                    user = (f"ROWS:\n" + "\n".join(parts)
                            + f"\n\nQUESTION: {q['question']}\n\nOne line: answer = ...")
                    raw = llm.complete(system=_CODEGEN_SYS, user=user,
                                       max_tokens=args.codegen_max_tokens)
                    val = None
                    try:
                        val = _safe_exec(_extract_code(raw))
                    except Exception:
                        val = None
                    if val is not None:
                        out_txt = (f"{val:.0f}" if float(val).is_integer()
                                   else f"{val:g}")
                    else:
                        out_txt = raw  # fell through; score whatever text came back
                    rec[arm]["used_codegen"] = int(val is not None)
                else:
                    user = (f"CONTEXT:\n" + "\n".join(parts)
                            + f"\n\nQUESTION: {q['question']}\n\nAnswer:")
                    out_txt = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=512)
                    if not out_txt and llm.last_finish_reason == "length":
                        out_txt = llm.complete(system=_DIRECT_SYS, user=user,
                                               max_tokens=1024)
                rec[arm]["answer_em"] = int(bool(
                    hitab_exact_match_text(out_txt, q["answer"])))
                rec[arm]["pred"] = out_txt[:120]
        recs.append(rec)
        rec_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        rec_fh.flush()
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(pop)}", flush=True)

    def agg(arm, key):
        return round(float(np.mean([r[arm][key] for r in recs])), 4)

    keys = ("osc", "per_cell", "gold_table_any", "gold_table_whole",
            "tokens", "n_tables") + (("answer_em",) if llm else ())
    summary = {a: {k: agg(a, k) for k in keys} for a in arms}
    paired = {}
    for a, b in (("cell", "dump"), ("cascade", "dump"), ("cell", "cascade"),
                 ("cell2dump", "dump"), ("cell2dump", "cell"),
                 # the head-to-head the paper needs: ours against the two units
                 # published table-RAG actually uses, all paying to find the table
                 ("cell", "row"), ("cell", "flat"), ("row", "dump"),
                 ("cellrow", "cell"), ("cellrow", "row"), ("cellrow", "dump"),
                 ("group", "cell"), ("group", "dump"), ("group", "row"),
                 ("capped", "cell"), ("capped", "group")):
        if a not in arms or b not in arms:
            continue
        for metric in ("osc",) + (("answer_em",) if llm else ()):
            paired[f"{metric}:{a}_vs_{b}"] = mcnemar(
                [r[a][metric] for r in recs], [r[b][metric] for r in recs])

    out = {
        "experiment": "corpus scale: retrieve-then-dump vs retrieve cells",
        "env": env,
        "population": {"name": args.population if args.dataset == "hitab" else
                       "aitqa, gold cells recovered by answer match, ambiguous dropped"
                       if args.dataset == "aitqa" else
                       f"realhitbench {'/'.join(args.rhb_subqtypes)}, gold cells "
                       "recovered by answer match, ambiguous dropped"
                       if args.dataset == "realhitbench" else
                       f"multihiertt table-only, seed {args.seed}, n={args.mh_queries}",
                       "n": len(recs),
                       "frozen": (str(pop_mod.path(args.population))
                                  if args.dataset == "hitab" else None)},
        "corpus": {"dataset": args.dataset, "split": args.split,
                   "tables": len(tids), "cells": len(cell_chunks)},
        "budget_tokens": args.budget, "budget_tokenizer": args.embed_model,
        "retriever": args.retriever, "alpha": alpha,
        "reader": llm.name if llm else None,
        # llm.name is only "local:<repo>" -- it drops the quantization and dtype
        # the spec carries, and those change the reader's output. Two runs paired
        # by query_id must share them, so record the spec verbatim rather than
        # leaving it to be inferred from when the run happened.
        "reader_spec": args.reader or None,
        "answer_mode": args.answer_mode,
        "cell_scheme": args.cell_scheme, "arms_run": list(arms),
        "max_tables": args.max_tables or None,
        "cell_title": not args.no_title, "table_index": args.table_index,
        "arms": {"dump": "table index -> whole tables in rank order while they fit",
                 "cell": "corpus-wide S2 cell index -> cells while they fit",
                 "cascade": "top-1 table by table index, then its cells only",
                 "cell2dump": "tables ranked by their best cell, dumped whole",
                 "cellrow": "rows ranked by their best cell, emitted as rows",
                 "row": "corpus-wide row-chunk index -> whole data rows",
                 "flat": "corpus-wide cell index, LEAF labels only (ablation)",
                 "capped": f"like `cell` but at most {args.cap} cells per table",
                 "group": "same cells and same ranking as `cell`, but rendered "
                          "grouped by table with the title stated ONCE -- the "
                          "title is 29% of every S3 sentence and is what the "
                          "gain comes from, so paying for it per table instead "
                          "of per cell buys back budget at no information loss"},
        "summary": summary, "paired_tests": paired,
        "table_recall": {
            "by_table_index": {f"@{k}": round(float(np.mean(
                [r["table_rank"] <= k for r in recs])), 4) for k in (1, 3, 5, 10, 20)},
            "by_cell_vote": {f"@{k}": round(float(np.mean(
                [r["table_rank_cellvote"] <= k for r in recs])), 4)
                for k in (1, 3, 5, 10, 20)}},
        "table_recall_paired": {
            f"@{k}": mcnemar([r["table_rank_cellvote"] <= k for r in recs],
                             [r["table_rank"] <= k for r in recs])
            for k in (1, 3, 5, 10, 20)},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    rec_fh.close()

    em_h = f"{'answer':>9}" if llm else ""
    print(f"\n{'arm':10}{'OSC':>8}{'per_cell':>10}{'tbl_any':>9}{'tbl_whole':>11}"
          f"{'tokens':>9}{'#tbl':>7}{em_h}")
    for a in arms:
        s = summary[a]
        print(f"{a:10}{s['osc']:>8.3f}{s['per_cell']:>10.3f}{s['gold_table_any']:>9.3f}"
              f"{s['gold_table_whole']:>11.3f}{s['tokens']:>9.0f}{s['n_tables']:>7.1f}"
              + (f"{s['answer_em']:>9.3f}" if llm else ""))
    print("\ntable recall  " + "".join(f"{f'@{k}':>9}" for k in (1, 3, 5, 10, 20)))
    for name in ("by_table_index", "by_cell_vote"):
        r = out["table_recall"][name]
        print(f"  {name:13}" + "".join(f"{r[f'@{k}']:>9.3f}" for k in (1, 3, 5, 10, 20)))
    for k, v in paired.items():
        print(f"  {k}: {v['only_first']}:{v['only_second']} p={v['exact_p']}")
    print(f"wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
