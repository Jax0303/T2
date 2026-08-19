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
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.bench import population as pop_mod
from rag_agent.bench.hitab import load_queries
from rag_agent.eval.metrics import hitab_exact_match_text
from rag_agent.generate.answerer import _DIRECT_SYS
from rag_agent.llm.factory import build_llm
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax
from rag_agent.runenv import guard_resume, run_env
from rag_agent.serialization.base import Chunk

from baseline_comparison_llm import Budget, markdown_table, row_chunks
from manual_sentence_ceiling import mcnemar
from point3_reconstruction_cost import build_table_paths, cell_text
from rag_agent.serialization.caption import caption_sentence

ARMS = ("dump", "cell", "cascade", "cell2dump", "row", "flat", "group", "capped")


class _CachedEncoder:
    """Memoize a corpus encoding on disk, keyed by the exact texts it covers.

    The key is a hash of the text list, so a corpus that changed for any reason
    -- a different split, a serialization fix, one more table -- misses the cache
    instead of silently reusing vectors that no longer describe it.
    """

    def __init__(self, inner, cache_dir: str, tag: str):
        self.inner, self.dir, self.tag = inner, Path(cache_dir), tag

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
        # the tag carries the embed model name, which has a "/" in it, so the
        # key is nested one level below cache_dir -- make the file's own parent
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


def table_index_text(raw, pt, bt) -> str:
    """What a TABLE-level retriever indexes: title/caption plus the header labels.

    Deliberately not the whole table body. A table-level index that contained
    every cell would be the cell index with extra steps, and no table retriever
    in the wild embeds a 3,000-token body into one vector.
    """
    title = " ".join(str(x) for x in (raw.get("title"), raw.get("caption")) if x)
    heads = {lab for d in ("gold_rp", "gold_cp") for p in pt[d].values()
             for lab in p if lab}
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


def hitab_corpus(data_dir: str, split: str, population: str) -> Corpus:
    queries, tables, paths, raws = build_corpus(data_dir, split)
    pop = [q for q in queries if q.gold_table_id in paths and q.gold_operands]
    pop = pop_mod.pin(population, pop) if population else pop
    tids = sorted(paths)
    md, ttext, shape, ctext, owner = {}, {}, {}, [], []
    ftext, rtext, rowner, titles, cpaths = [], [], [], {}, []
    for tid in tids:
        raw, pt, bt = raws[tid], paths[tid], tables[tid]
        titles[tid] = " ".join(str(x) for x in
                               (raw.get("title"), raw.get("caption")) if x)
        md[tid] = markdown_table(raw, max(1, len(raw["texts"]) - pt["n_r"]))
        ttext[tid] = table_index_text(raw, pt, bt)
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


def aitqa_corpus(data_dir: str = "data/aitqa") -> Corpus:
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
    return Corpus(tids, md, ttext, shape, ctext, owner, qs, ftext, rtext,
                  rowner, {t: '' for t in tids}, cpaths)


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
                    choices=["hitab", "multihiertt", "aitqa"])
    ap.add_argument("--mh-queries", type=int, default=400,
                    help="MultiHiertt questions to sample (their documents become "
                         "the corpus)")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_arith")
    ap.add_argument("--budget", type=int, default=1024, help="context tokens per arm")
    ap.add_argument("--retriever", default="bm25", choices=list(ALPHA))
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell",
                    help="where corpus embeddings are memoized across budgets")
    ap.add_argument("--reader", default="",
                    help="answer the questions too, with this model (e.g. "
                         "local:Qwen/Qwen2.5-7B-Instruct). Every arm gets the same "
                         "budget and the same prompt frame, so an answer-EM "
                         "difference is a difference in what the budget bought")
    ap.add_argument("--cell-scheme", default="S2", choices=["S2", "S3"],
                    help="what the CELL arm indexes. S2 is the bare header path "
                         "('a > b > c: v'); S3 is this work's deployed index unit "
                         "-- a sentence stating the table title and both paths "
                         "(rag_agent/serialization/caption.py). Only the cell and "
                         "cascade arms change; dump/row/flat do not read it")
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
    out_path = args.out or f"results/corpus_dump_vs_cell_{args.dataset}_{args.retriever}_{args.budget}.json"

    if args.dataset == "hitab":
        C = hitab_corpus(args.data_dir, args.split, args.population)
    elif args.dataset == "aitqa":
        C = aitqa_corpus()
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
    if args.cell_scheme == "S3":
        # re-render from the same paths the S2 text was built from, so the two
        # schemes differ in rendering only and the cell SET stays identical
        C.cell_text[:] = [
            caption_sentence("" if args.no_title else C.title.get(t, ""),
                             C.cell_paths[n][0],
                             C.cell_paths[n][1], C.cell_paths[n][2])
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

    # what the grouped arm renders: one header per table, one short line per cell
    grp_head = {t: f"[table] {C.title.get(t, '')}".rstrip() for t in tids}
    grp_line = [f"  {' > '.join(rp)} | {' > '.join(cp)} : {v}"
                for rp, cp, v in C.cell_paths]
    grp_head_tok = {t: bud.count(x) + 1 for t, x in grp_head.items()}
    grp_line_tok = np.array([bud.count(x) for x in grp_line], dtype=np.int32)

    cells_by_table: dict[str, list[int]] = {}
    for n, (tid, _, _) in enumerate(cell_owner):
        cells_by_table.setdefault(tid, []).append(n)

    alpha = ALPHA[args.retriever]
    tok_cache: dict[str, int] = {}

    def md_tokens(tid: str) -> int:
        if tid not in tok_cache:
            tok_cache[tid] = bud.count("\n".join(C.md_lines[tid]))
        return tok_cache[tid]

    rec_path = Path(str(Path(out_path).with_suffix("")) + "_records.jsonl")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    guard_resume(rec_path, env, reader=(llm.name if llm else None),
                 population=(args.population if args.dataset == "hitab"
                             else "aitqa_answer_matched" if args.dataset == "aitqa"
                             else f"multihiertt_{args.mh_queries}_{args.seed}"),
                 force=args.force_resume)
    rec_fh = open(rec_path, "w")

    recs = []
    for n_done, q in enumerate(pop, 1):
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
                       f"multihiertt table-only, seed {args.seed}, n={args.mh_queries}",
                       "n": len(recs),
                       "frozen": (str(pop_mod.path(args.population))
                                  if args.dataset == "hitab" else None)},
        "corpus": {"dataset": args.dataset, "split": args.split,
                   "tables": len(tids), "cells": len(cell_chunks)},
        "budget_tokens": args.budget, "budget_tokenizer": args.embed_model,
        "retriever": args.retriever, "reader": llm.name if llm else None,
        "cell_scheme": args.cell_scheme, "arms_run": list(arms),
        "cell_title": not args.no_title,
        "arms": {"dump": "table index -> whole tables in rank order while they fit",
                 "cell": "corpus-wide S2 cell index -> cells while they fit",
                 "cascade": "top-1 table by table index, then its cells only",
                 "cell2dump": "tables ranked by their best cell, dumped whole",
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
