#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bottleneck diagnosis for the k-ladder finding (K_LADDER_TABLES-2026-09-16.md):
retrieval recall rises with k while answer accuracy given retrieval success falls
(distractor dilution), and overall answer accuracy rises only slowly. This adds
four diagnostics on TOP of that table, reusing its data and code instead of
recomputing anything already on disk:

  rankbucket    per-query gold_rank (already in s3c_v2_records.jsonl, computed
                by scripts/retrieval_accuracy.py) bucketed into
                [1,2,3,4-5,6-10,11-20,>20], N and answer accuracy per bucket for
                each k already run (k1/k5/k10/k20 legs in
                results/k_ladder_qwen3_8b_20260916/). No new retrieval, no new
                reader calls.
  errorclass    when the top-1 retrieved cell (context_units[0], already dumped
                by retrieval_accuracy.py) is not the gold cell, classify the
                relationship (same_value/wrong_row/wrong_column/
                same_leaf_header/nearby_cell/wrong_table/other). No new
                retrieval, no reader calls.
  sample        fixed stratified sample (by the same rank buckets, proportional
                allocation, seed-fixed) for the controlled-QA legs below.
  controlledqa  one leg = one (distractor_count, distractor_type, gold_position)
                condition, run over the `sample` population. distractor pools:
                hard_negative reuses context_units (already-ranked hybrid score,
                no new embedding/BM25 call); random and same_value scan the
                query's own table (hg.load_table, already cached elsewhere).
                Same resume/fsync/--out contract as scripts/answer_accuracy.py
                (reuses its PROMPTS, load_saved_rows, check_context_limit).
  report        assemble whatever of the above has been written into one MD.

ESM/Recall/Precision/F1 definitions and names are NOT touched — this script
only re-buckets a rank that scripts/retrieval_accuracy.py already computed.

  PYTHONPATH=. .venv/bin/python scripts/bottleneck_diagnosis.py rankbucket
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_diagnosis.py errorclass
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_diagnosis.py structure
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_diagnosis.py sample --n 150
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_diagnosis.py controlledqa \\
      --condition gold_1_random_first --resume
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_diagnosis.py report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.metrics import hitab_exact_match_text             # noqa: E402
from rag_agent.eval.artifacts import (digest, read_records,           # noqa: E402
                                     require_same_ids, validate_retrieval)
from rag_agent.llm.factory import build_llm                           # noqa: E402
from rag_agent.serialization.caption import caption_sentence, with_page_title  # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT      # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES                    # noqa: E402
from scripts.answer_accuracy import (PROMPTS, check_context_limit,    # noqa: E402
                                     load_saved_rows)

RECORDS = ROOT / "results/evaluation_v2/s3c_v2_records.jsonl"
K_LADDER_DIR = ROOT / "results/k_ladder_qwen3_8b_20260916"
K_LEGS = {1: "k1", 5: "k5", 10: "k10", 20: "k20"}
OUT_DIR = ROOT / "results/bottleneck_diagnosis"
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def load_primary_population() -> dict:
    """The 991-query preregistered single-cell population, keyed by query_id.

    Same filter scripts/answer_accuracy.py --primary-only applies: mode=all,
    m=1, aggregation=none.
    """
    records = read_records(RECORDS)
    for row in records.values():
        validate_retrieval(row)
    return {qid: r for qid, r in records.items()
            if "correct" in r and r.get("mode") == "all" and r.get("m") == 1
            and (r.get("aggregation") or "none") == "none"}


def split_corpus_table_ids() -> list:
    """The 538 tables the split corpus indexes (evaluation_v2/s3c_v2 ``corpus:
    split``) — every ``table_id`` any record (scored or excluded) names."""
    records = read_records(RECORDS)
    return sorted({r["table_id"] for r in records.values() if "table_id" in r})


VALUE_INDEX_CACHE = OUT_DIR / "value_index_cache.json"


def build_value_index(data_dir: str = "data/hitab") -> dict:
    """``norm_value -> [(table_id, i, j), ...]`` over the whole split corpus.

    A per-table pool for same-value distractors starved almost every query
    (measured on the n=150 controlled-QA sample: count=1 43/150, count=4
    4/150, count=9 1/150 queries had enough same-table duplicates) — most
    HiTab tables just don't repeat a value 9 times. Corpus-wide is the
    pool that is actually usable, and it is still no dataset/ML cost: one
    read of the 538 tables the split already indexes, cached to disk since
    every controlledqa leg runs as its own process.
    """
    if VALUE_INDEX_CACHE.exists():
        raw = json.loads(VALUE_INDEX_CACHE.read_text())
        return {v: [tuple(c) for c in coords] for v, coords in raw.items()}
    index = defaultdict(list)
    for tid in split_corpus_table_ids():
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = str(t.data[i][j]).strip()
                if v:
                    index[hg.norm_value(v)].append((tid, i, j))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VALUE_INDEX_CACHE.write_text(json.dumps(index, ensure_ascii=False))
    return dict(index)


def load_answer_leg(k: int) -> dict:
    path = K_LADDER_DIR / f"s3c_v2_primary_qwen3_8b_{K_LEGS[k]}.jsonl"
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            out[row["query_id"]] = row
    return out


# ---------------------------------------------------------------------------
# rankbucket
# ---------------------------------------------------------------------------

RANK_BUCKET_LABELS = ("1", "2", "3", "4-5", "6-10", "11-20", ">20")


def rank_bucket(rank) -> str:
    """``gold_rank`` (1-based, ``None`` if not found in the top-500 scan
    scripts/retrieval_accuracy.py does) -> one of RANK_BUCKET_LABELS."""
    if rank is None or rank > 20:
        return ">20"
    if rank <= 3:
        return str(rank)
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    return "11-20"


def rank_buckets_table() -> dict:
    pop = load_primary_population()
    buckets = {qid: rank_bucket(r.get("gold_rank")) for qid, r in pop.items()}
    legs = {k: load_answer_leg(k) for k in K_LEGS}
    for k, leg in legs.items():
        require_same_ids(pop, leg, label=f"k{k} answer leg")

    rows = []
    for label in RANK_BUCKET_LABELS:
        qids = [qid for qid, b in buckets.items() if b == label]
        row = {"rank_bucket": label, "n": len(qids)}
        for k, leg in legs.items():
            vals = [leg[qid]["answer_correct"] for qid in qids]
            row[f"qa_k{k}"] = round(sum(vals) / len(vals), 4) if vals else None
        rows.append(row)
    n_none = sum(1 for r in pop.values() if r.get("gold_rank") is None)
    return {"population": "hitab test primary (mode=all, m=1, aggregation=none)",
            "n": len(pop), "n_gold_rank_none_in_top500": n_none,
            "note": "N is the same across k (retrieval-side quantity); qa_k{K} is "
                    "answer accuracy of the k-leg's own answer file for the "
                    "queries in that bucket.",
            "rows": rows}


# ---------------------------------------------------------------------------
# errorclass
# ---------------------------------------------------------------------------

ERROR_CLASSES = ("same_value", "wrong_row", "wrong_column", "same_leaf_header",
                 "nearby_cell", "wrong_table", "other")


def classify_top1_error(pred, gold, tabs, data_dir="data/hitab") -> str:
    """Relationship between the top-1 retrieved cell ``pred`` and the single
    gold cell ``gold`` (both ``(table_id, i, j)``). Checked in priority order;
    first match wins:

      same_value        predicted cell's value equals gold's value, wherever
                         it sits (the reader could still answer correctly from
                         it despite the retrieval miss).
      wrong_row         same table, same column, different row.
      wrong_column      same table, same row, different column.
      same_leaf_header  same table, different row AND column, but shares a
                         leaf row or column label with gold (an ambiguous
                         repeated header, e.g. "Total" under two parents).
      nearby_cell       same table, Chebyshev distance 1 (an off-by-one
                         neighbour), none of the above.
      wrong_table       different table entirely, no value match.
      other             same table, none of the above.
    """
    ptid, pi, pj = pred
    gtid, gi, gj = gold

    def tab(tid):
        t = tabs.get(tid)
        if t is None:
            t = tabs[tid] = hg.load_table(tid, data_dir)
        return t

    pt, gt = tab(ptid), tab(gtid)
    if pt is None or gt is None:
        return "other"
    pv = hg.norm_value(pt.table.data[pi][pj])
    gv = hg.norm_value(gt.table.data[gi][gj])
    if pv == gv and (ptid, pi, pj) != (gtid, gi, gj):
        return "same_value"
    if ptid != gtid:
        return "wrong_table"
    if pj == gj and pi != gi:
        return "wrong_row"
    if pi == gi and pj != gj:
        return "wrong_column"
    p_row_leaf = pt.table.row_path(pi)[-1] if pt.table.row_path(pi) else None
    p_col_leaf = pt.table.col_path(pj)[-1] if pt.table.col_path(pj) else None
    g_row_leaf = gt.table.row_path(gi)[-1] if gt.table.row_path(gi) else None
    g_col_leaf = gt.table.col_path(gj)[-1] if gt.table.col_path(gj) else None
    if (p_row_leaf is not None and p_row_leaf == g_row_leaf) or \
       (p_col_leaf is not None and p_col_leaf == g_col_leaf):
        return "same_leaf_header"
    if max(abs(pi - gi), abs(pj - gj)) <= 1:
        return "nearby_cell"
    return "other"


def error_classification_table(data_dir="data/hitab") -> dict:
    pop = load_primary_population()
    tabs: dict = {}
    counts = Counter()
    examples = defaultdict(list)
    n_top1_wrong = 0
    for qid, r in pop.items():
        units = r["context_units"]
        if not units:
            continue
        pred = tuple(units[0]["cells"][0])
        gold = tuple(sorted(r["gold_cells"])[0])
        if pred == gold:
            continue
        n_top1_wrong += 1
        cls = classify_top1_error(pred, gold, tabs, data_dir)
        counts[cls] += 1
        if len(examples[cls]) < 5:
            examples[cls].append({"query_id": qid, "predicted_cell": list(pred),
                                  "gold_cell": list(gold)})
    return {"population": "hitab test primary (mode=all, m=1, aggregation=none)",
            "n": len(pop), "n_top1_correct": len(pop) - n_top1_wrong,
            "n_top1_wrong": n_top1_wrong,
            "classes": [{"class": c, "n": counts[c],
                        "share_of_wrong": round(counts[c] / n_top1_wrong, 4) if n_top1_wrong else None,
                        "examples": examples[c]} for c in ERROR_CLASSES]}


# ---------------------------------------------------------------------------
# structure comparison (item 4) — skip if the data does not exist
# ---------------------------------------------------------------------------

def structure_comparison() -> dict:
    """predicted-vs-gold STRUCTURE retrieval (table/header prediction) Recall@k,
    MRR, median rank. The table-first retrieval line that would have produced
    predicted structure was discarded and its files deleted (STAGE 5,
    9bf1586); nothing in the current tree predicts table/header structure
    ahead of cell retrieval, so this is a documented skip, not a null result.
    """
    candidates = list((ROOT / "results").glob("**/*structure_pred*"))
    if candidates:
        return {"skipped": False, "candidate_files": [str(p) for p in candidates]}
    return {"skipped": True,
            "reason": "no predicted-structure retrieval output in the current tree "
                     "(the table-first retrieval line, STAGE 5, was discarded and "
                     "its files deleted 2026-09-15/16; recoverable only from "
                     "git show 9bf1586, not restored here without a user instruction)"}


# ---------------------------------------------------------------------------
# sample — fixed stratified sample for controlled QA
# ---------------------------------------------------------------------------

def stratified_sample(n: int, seed: int) -> list:
    pop = load_primary_population()
    buckets = defaultdict(list)
    for qid, r in pop.items():
        buckets[rank_bucket(r.get("gold_rank"))].append(qid)
    for qids in buckets.values():
        qids.sort()                                    # deterministic before shuffling

    # Largest-remainder allocation so the sample matches the bucket shares of
    # the full population and sums to exactly n.
    sizes = {b: len(buckets.get(b, [])) for b in RANK_BUCKET_LABELS}
    total = sum(sizes.values())
    raw = {b: sizes[b] * n / total for b in RANK_BUCKET_LABELS}
    alloc = {b: int(raw[b]) for b in RANK_BUCKET_LABELS}
    remainder = n - sum(alloc.values())
    for b in sorted(RANK_BUCKET_LABELS, key=lambda b: raw[b] - alloc[b], reverse=True)[:remainder]:
        alloc[b] += 1

    out = []
    for b in RANK_BUCKET_LABELS:
        rng = random.Random(int(digest({"seed": seed, "bucket": b})[:8], 16))
        out += rng.sample(buckets.get(b, []), min(alloc[b], len(buckets.get(b, []))))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# controlled QA
# ---------------------------------------------------------------------------

DISTRACTOR_COUNTS = (1, 4, 9)
DISTRACTOR_TYPES = ("random", "hard_negative", "same_value")
GOLD_POSITIONS = ("first", "last")


def condition_names() -> list:
    names = ["gold_only"]
    for c in DISTRACTOR_COUNTS:
        for t in DISTRACTOR_TYPES:
            for p in GOLD_POSITIONS:
                names.append(f"gold_{c}_{t}_{p}")
    return names


def parse_condition(name: str):
    if name == "gold_only":
        return None
    _, count, *rest, pos = name.split("_")
    return int(count), "_".join(rest), pos


def cell_sentence(tab, i, j) -> str:
    t = tab.table
    title = with_page_title(tab.title, _PAGE_TITLES.get(tab.table_id))
    return caption_sentence(title, t.row_path(i), t.col_path(j),
                            value=t.data[i][j], template=STRUCTURAL_COMPACT)


def render_coord(coord, tabs: dict, data_dir: str) -> str:
    tid, i, j = coord
    tab = tabs.get(tid)
    if tab is None:
        tab = tabs[tid] = hg.load_table(tid, data_dir)
    return cell_sentence(tab, i, j)


def pick_random(tab, exclude, count: int, seed: int) -> list:
    t = tab.table
    live = [(i, j) for i in range(t.n_rows) for j in range(t.n_cols)
            if str(t.data[i][j]).strip() and (i, j) != exclude]
    live.sort()
    rng = random.Random(seed)
    return [(tab.table_id, i, j) for i, j in rng.sample(live, min(count, len(live)))]


def pick_hard_negative(units, gold, count: int) -> list:
    """Distractors from the query's OWN already-ranked hybrid retrieval
    (``context_units``, corpus=split — so a candidate may belong to a
    DIFFERENT table than gold; that is what makes it a realistic hard
    negative, not a same-table-only one)."""
    out = []
    for u in units:
        c = tuple(u["cells"][0])
        if c == gold:
            continue
        out.append(c)
        if len(out) == count:
            break
    return out


def pick_same_value(value_index: dict, exclude, gold_value, count: int) -> list:
    """Corpus-wide (split, 538 tables) same-normalised-value cells, gold
    excluded. See build_value_index for why this is corpus-wide, not
    per-table."""
    gv = hg.norm_value(gold_value)
    cand = sorted(c for c in value_index.get(gv, []) if c != exclude)
    return cand[:count]


def build_condition_context(name: str, r: dict, tabs: dict, data_dir: str, value_index: dict):
    """``(context_lines, dropped_reason)``. dropped_reason is set (and
    context_lines is []) when the query cannot support this condition (e.g.
    same_value needs more duplicate-valued cells than the corpus has)."""
    gold = tuple(sorted(r["gold_cells"])[0])
    tid, gi, gj = gold
    tab = tabs.get(tid) or hg.load_table(tid, data_dir)
    tabs[tid] = tab
    gold_line = cell_sentence(tab, gi, gj)
    parsed = parse_condition(name)
    if parsed is None:
        return [gold_line], None
    count, dtype, pos = parsed
    if dtype == "random":
        seed = int(digest({"cond": name, "query_id": r["query_id"]})[:8], 16)
        coords = pick_random(tab, (gi, gj), count, seed)
    elif dtype == "hard_negative":
        coords = pick_hard_negative(r["context_units"], gold, count)
    elif dtype == "same_value":
        coords = pick_same_value(value_index, gold, tab.table.data[gi][gj], count)
    else:
        raise ValueError(f"unknown distractor type: {dtype}")
    if len(coords) < count:
        return [], f"insufficient_{dtype}_candidates ({len(coords)}/{count})"
    distractor_lines = [render_coord(c, tabs, data_dir) for c in coords]
    lines = [gold_line, *distractor_lines] if pos == "first" else [*distractor_lines, gold_line]
    return lines, None


def run_controlled_qa(condition: str, sample_ids: list, out: Path, reader: str,
                      data_dir: str, max_tokens: int, seed: int, resume: bool) -> dict:
    pop = load_primary_population()
    missing = [qid for qid in sample_ids if qid not in pop]
    if missing:
        raise ValueError(f"{len(missing)} sample query_ids are not in the primary population")
    order = list(sample_ids)
    tabs: dict = {}
    value_index = build_value_index(data_dir) if "same_value" in condition else {}
    contexts, dropped = {}, {}
    for qid in order:
        lines, reason = build_condition_context(condition, pop[qid], tabs, data_dir, value_index)
        if reason:
            dropped[qid] = reason
        else:
            contexts[qid] = lines
    order = [qid for qid in order if qid in contexts]
    if not order:
        raise ValueError(f"{condition}: every sampled query was dropped ({dropped})")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix(".json").exists():
        raise SystemExit(f"{out.with_suffix('.json')} exists — this leg is done; pick a new --out to redo it")
    if out.exists() and not resume:
        raise SystemExit(f"{out} has a partial run — pass --resume (never overwritten)")

    llm = build_llm(reader)
    import torch                                                      # noqa: E402
    torch.manual_seed(seed)
    limit = getattr(llm, "context_limit", 0)
    done = load_saved_rows(out, order) if out.exists() else {}
    rows, t0 = list(done.values()), time.time()
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        for k, qid in enumerate(order, 1):
            if qid in done:
                continue
            r = pop[qid]
            ctx = contexts[qid]
            user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {r['question']}\nAnswer:"
            n_tok = (llm.n_prompt_tokens(PROMPTS["neutral"], user)
                     if hasattr(llm, "n_prompt_tokens") else None)
            check_context_limit(n_tok, max_tokens, limit)
            pred = llm.complete(PROMPTS["neutral"], user, max_tokens=max_tokens, temperature=0.0)
            ok = hitab_exact_match_text(pred, r["answer"])
            row = {"query_id": qid, "condition": condition, "n_ctx": len(ctx), "n_tok": n_tok,
                  "answer_correct": int(ok), "pred": pred, "answer": r["answer"]}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
            if k % 25 == 0:
                print(f"  [{condition}] {k}/{len(order)}  {time.time() - t0:.0f}s  "
                      f"acc={sum(x['answer_correct'] for x in rows) / len(rows):.4f}", flush=True)
    if [x["query_id"] for x in rows] != order:
        raise SystemExit(f"{condition}: saved rows differ from the run plan")

    acc = round(sum(x["answer_correct"] for x in rows) / len(rows), 4)
    summary = {"condition": condition, "reader": llm.name, "seed": seed,
              "max_new_tokens": max_tokens, "n_sampled": len(sample_ids),
              "n_dropped": len(dropped), "dropped_reasons": dict(Counter(dropped.values())),
              "n": len(rows), "answer_accuracy": acc,
              "records_sha256": digest(rows)}
    with out.with_suffix(".json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def write_md() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# 병목 진단 (2026-09-16)", "",
             "K_LADDER_TABLES-2026-09-16.md 의 k-사다리 현상(recall 상승 / 검색성공시 "
             "정확도 하락 / 전체 정확도 완만 상승)을 더 세분화한 진단. 기존 ESM/Recall/"
             "Precision/F1 정의·명칭은 바꾸지 않는다.", ""]

    rb_path = OUT_DIR / "rank_buckets.json"
    if rb_path.exists():
        d = json.loads(rb_path.read_text())
        lines += ["## 1. gold-cell rank 버킷별 N / 답변 정확도", "",
                  f"모집단: {d['population']} (query count={d['n']}, gold_rank 미확인 {d['n_gold_rank_none_in_top500']}건 "
                  "→ '>20' 버킷에 포함)", "",
                  "| rank | N | QA k=1 | QA k=5 | QA k=10 | QA k=20 |",
                  "|---|---|---|---|---|---|"]
        for row in d["rows"]:
            lines.append(f"| {row['rank_bucket']} | {row['n']} | "
                         f"{row['qa_k1']} | {row['qa_k5']} | {row['qa_k10']} | {row['qa_k20']} |")
        lines.append("")

    ec_path = OUT_DIR / "top1_error_classification.json"
    if ec_path.exists():
        d = json.loads(ec_path.read_text())
        lines += ["## 2. top-1 검색 오류 분류", "",
                  f"모집단: {d['population']} (query count={d['n']}, top-1 정답 {d['n_top1_correct']}건, "
                  f"top-1 오답 {d['n_top1_wrong']}건)", "",
                  "| class | N | 오답 중 비율 |", "|---|---|---|"]
        for c in d["classes"]:
            lines.append(f"| {c['class']} | {c['n']} | {c['share_of_wrong']} |")
        lines.append("")

    cq_dir = OUT_DIR
    legs = sorted(cq_dir.glob("controlled_qa_gold_*.json"))
    if legs:
        lines += ["## 3. Controlled QA (gold + 통제된 distractor)", "",
                  "| condition | query count | dropped | 답변 정확도 |", "|---|---|---|---|"]
        for p in legs:
            d = json.loads(p.read_text())
            lines.append(f"| {d['condition']} | {d['n']} | {d['n_dropped']} | {d['answer_accuracy']} |")
        lines.append("")

    st_path = OUT_DIR / "structure_comparison.json"
    if st_path.exists():
        d = json.loads(st_path.read_text())
        lines += ["## 4. predicted-vs-gold structure retrieval", ""]
        lines.append("skip — " + d["reason"] if d.get("skipped") else json.dumps(d, ensure_ascii=False))
        lines.append("")

    out = OUT_DIR / "BOTTLENECK_DIAGNOSIS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("rankbucket")
    sub.add_parser("errorclass")
    sub.add_parser("structure")

    sp = sub.add_parser("sample")
    sp.add_argument("--n", type=int, default=150)
    sp.add_argument("--seed", type=int, default=42)

    cq = sub.add_parser("controlledqa")
    cq.add_argument("--condition", required=True, choices=condition_names())
    cq.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit")
    cq.add_argument("--data-dir", default="data/hitab")
    cq.add_argument("--max-tokens", type=int, default=64)
    cq.add_argument("--seed", type=int, default=42)
    cq.add_argument("--resume", action="store_true")
    cq.add_argument("--out", default="")

    sub.add_parser("report")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.cmd == "rankbucket":
        d = rank_buckets_table()
        (OUT_DIR / "rank_buckets.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(d, indent=2, ensure_ascii=False))
    elif a.cmd == "errorclass":
        d = error_classification_table()
        (OUT_DIR / "top1_error_classification.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in d.items() if k != "classes"}, indent=2, ensure_ascii=False))
    elif a.cmd == "structure":
        d = structure_comparison()
        (OUT_DIR / "structure_comparison.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(d, indent=2, ensure_ascii=False))
    elif a.cmd == "sample":
        ids = stratified_sample(a.n, a.seed)
        p = OUT_DIR / "controlled_qa_sample.json"
        if p.exists():
            raise SystemExit(f"{p} exists — the sample is fixed once drawn; delete it explicitly to redraw")
        p.write_text(json.dumps({"n": len(ids), "seed": a.seed, "query_ids": ids},
                                indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(ids)} query_ids -> {p}")
    elif a.cmd == "controlledqa":
        sample_path = OUT_DIR / "controlled_qa_sample.json"
        if not sample_path.exists():
            raise SystemExit(f"run `sample` first ({sample_path} missing)")
        sample_ids = json.loads(sample_path.read_text())["query_ids"]
        out = Path(a.out or OUT_DIR / f"controlled_qa_{a.condition}.jsonl")
        if out.suffix != ".jsonl":
            raise SystemExit("--out must end in .jsonl")
        run_controlled_qa(a.condition, sample_ids, out, a.reader, a.data_dir,
                          a.max_tokens, a.seed, a.resume)
    elif a.cmd == "report":
        p = write_md()
        print(f"wrote -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
