#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pre-improvement diagnostic extraction (results/pre_improvement_audit/).

Read-only: no model/encoder/reader calls, no ranking or existing-file changes.
Everything here reuses artifacts and pure local computation that already sit
on disk:

  - results/evaluation_v2/s3c_v2_records.jsonl   the deployed top-20 per query
    (context_units: cell + serialized text, already ranked -- position in the
    list IS the rank; no scores stored)
  - results/bottleneck_root_cause/retrieval_gold_path.jsonl   per-query
    gold/top1 dense+hybrid scores for the primary (m=1) population, computed
    against the SAME corpus (byte-identical text -> cache hit, no re-encode)
  - rag_agent.bench.hitab_grid.load_table   table grids parsed straight from
    data/hitab/*.json (no model)
  - rag_agent.reconstruct header reconstruction (H1 predicted structure) --
    deterministic algorithm over the raw grid, no model

Per-candidate dense_score is only available where the candidate IS the query's
gold cell or top-1 cell (that is all any existing run persisted); every other
rank's score is null with an explicit reason -- recomputing it needs a fresh
query-encoder call, which this script refuses to make.

  PYTHONPATH=. .venv/bin/python scripts/pre_improvement_audit.py all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.artifacts import digest, read_records             # noqa: E402
from rag_agent.serialization.caption import with_page_title           # noqa: E402
from retrieval_accuracy import PAGE_TITLES, cell_unit                 # noqa: E402
from bottleneck_diagnosis import (classify_top1_error,                # noqa: E402
                                  load_primary_population,
                                  split_corpus_table_ids)
from bottleneck_root_cause import (REPRESENTATIONS, build_repr_corpus,  # noqa: E402
                                   cell_structure_correct, load_retrieval_rows,
                                   predicted_structure, unknown_rank_query_ids)

DATA_DIR = "data/hitab"
RECORDS_PATH = ROOT / "results/evaluation_v2/s3c_v2_records.jsonl"
OUT_DIR = ROOT / "results/pre_improvement_audit"
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def get_table(tid, cache):
    t = cache.get(tid)
    if t is None and tid not in cache:
        t = hg.load_table(tid, DATA_DIR)
        cache[tid] = t
    return t


def title_for(tid, cache):
    tab = get_table(tid, cache)
    return with_page_title(tab.title, _PAGE_TITLES.get(tid)) if tab else None


def cell_paths(tid, i, j, cache):
    """``(row_path, col_path, value)`` or ``(None, None, None)`` if the table
    is unloadable or the coordinate is out of range."""
    tab = get_table(tid, cache)
    if tab is None:
        return None, None, None
    t = tab.table
    if not (0 <= i < t.n_rows and 0 <= j < t.n_cols):
        return None, None, None
    return t.row_path(i), t.col_path(j), t.data[i][j]


def serialize_cell(tid, i, j, cache):
    rp, cp, v = cell_paths(tid, i, j, cache)
    if rp is None:
        return None
    title = title_for(tid, cache)
    return cell_unit(title, rp, cp, v, "s3c")


def predicted_paths(tid, i, j, cache):
    row_fn, col_fn, meta = predicted_structure(tid, DATA_DIR)
    if row_fn is None:
        return None, None, meta
    return row_fn(i), col_fn(j), meta


# ---------------------------------------------------------------------------
# item 1 -- top20_detail
# ---------------------------------------------------------------------------

def gold_rank_status(gold_rank):
    if gold_rank is None:
        return "unknown_beyond_scan"
    if gold_rank <= 20:
        return "in_top20"
    return "rank_21_500"


def build_top20_rows(records: dict, gold_path_rows: dict, tabs: dict) -> list:
    rows = []
    for qid, r in sorted(records.items()):
        if "correct" not in r:
            rows.append({
                "query_id": qid, "excluded_reason": r.get("excluded"),
                "table_id": r.get("table_id"), "mode": r.get("mode"),
                "rank": None,
            })
            continue
        gold_cells = [tuple(c) for c in r["gold_cells"]]
        gold_ref = sorted(gold_cells)[0]
        gtid, gi, gj = gold_ref
        grp, gcp, gval = cell_paths(gtid, gi, gj, tabs)
        gtext = serialize_cell(gtid, gi, gj, tabs)
        g_pred_rp, g_pred_cp, g_struct_meta = predicted_paths(gtid, gi, gj, tabs)
        g_struct_ok = cell_structure_correct(gtid, gi, gj, DATA_DIR)
        gp_row = gold_path_rows.get(qid)
        common = {
            "query_id": qid, "query": r.get("question"),
            "mode": r["mode"], "m": r["m"], "aggregation": r.get("aggregation"),
            "n_gold_cells": len(gold_cells),
            "gold_table": gtid, "gold_row": gi, "gold_col": gj,
            "gold_row_path": list(grp) if grp is not None else None,
            "gold_col_path": list(gcp) if gcp is not None else None,
            "gold_value": gval, "gold_serialized_text": gtext,
            "gold_predicted_row_path": list(g_pred_rp) if g_pred_rp is not None else None,
            "gold_predicted_col_path": list(g_pred_cp) if g_pred_cp is not None else None,
            "gold_structure_correct": g_struct_ok,
            "table_caption": title_for(gtid, tabs),
            "gold_rank": r.get("gold_rank"),
            "gold_rank_status": gold_rank_status(r.get("gold_rank")),
            "gold_table_in_context": r.get("gold_table_in_context"),
            "dense_score_gold": gp_row["dense_score_gold"] if gp_row else None,
            "hybrid_score_gold": gp_row["hybrid_score_gold"] if gp_row else None,
            "score_source": ("bottleneck_root_cause/retrieval_gold_path.jsonl"
                             if gp_row else
                             "not available: score backfill only exists for the "
                             "primary single-cell population (m=1, mode=all, "
                             "aggregation=none)"),
        }
        units = r.get("context_units") or []
        if not units:
            rows.append({**common, "rank": None, "candidate_note": "no context_units delivered"})
            continue
        for k, u in enumerate(units, 1):
            ctid, ci, cj = tuple(u["cells"][0])
            crp, ccp, cval = cell_paths(ctid, ci, cj, tabs)
            c_pred_rp, c_pred_cp, _ = predicted_paths(ctid, ci, cj, tabs)
            c_struct_ok = cell_structure_correct(ctid, ci, cj, DATA_DIR)
            is_gold = (ctid, ci, cj) in gold_cells
            if (ctid, ci, cj) == gold_ref:
                err = "exact_match"
            elif is_gold:
                err = "other_gold_cell"
            else:
                err = classify_top1_error((ctid, ci, cj), gold_ref, tabs, DATA_DIR)
            dense = hybrid = None
            score_note = ("not persisted: only the query's gold cell and rank-1 "
                          "cell have a saved dense/hybrid score (see score_source "
                          "above); recomputing this rank needs a fresh query-"
                          "encoder call, excluded per no-new-API-call instruction")
            if gp_row:
                if (ctid, ci, cj) == tuple(gp_row["gold_cell"]):
                    dense, hybrid, score_note = gp_row["dense_score_gold"], gp_row["hybrid_score_gold"], None
                elif (ctid, ci, cj) == tuple(gp_row["top1_cell"]):
                    dense, hybrid, score_note = gp_row["dense_score_top1"], gp_row["hybrid_score_top1"], None
            rows.append({
                **common, "rank": k,
                "cand_table": ctid, "cand_row": ci, "cand_col": cj,
                "cand_row_path": list(crp) if crp is not None else None,
                "cand_col_path": list(ccp) if ccp is not None else None,
                "cand_value": cval,
                "cand_serialized_text": u["text"],
                "cand_predicted_row_path": list(c_pred_rp) if c_pred_rp is not None else None,
                "cand_predicted_col_path": list(c_pred_cp) if c_pred_cp is not None else None,
                "cand_structure_correct": c_struct_ok,
                "is_gold": is_gold, "error_class": err,
                "dense_score": dense, "hybrid_score": hybrid,
                "score_note": score_note,
            })
    return rows


CSV_FIELDS = [
    "query_id", "query", "mode", "m", "aggregation", "n_gold_cells",
    "gold_table", "gold_row", "gold_col", "gold_row_path", "gold_col_path",
    "gold_value", "gold_serialized_text", "gold_predicted_row_path",
    "gold_predicted_col_path", "gold_structure_correct", "table_caption",
    "gold_rank", "gold_rank_status", "gold_table_in_context",
    "dense_score_gold", "hybrid_score_gold", "score_source",
    "rank", "cand_table", "cand_row", "cand_col", "cand_row_path",
    "cand_col_path", "cand_value", "cand_serialized_text",
    "cand_predicted_row_path", "cand_predicted_col_path",
    "cand_structure_correct", "is_gold", "error_class",
    "dense_score", "hybrid_score", "score_note",
    "excluded_reason", "table_id", "candidate_note",
]


def write_top20(rows: list) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "top20_detail.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def flat(v):
        if isinstance(v, list):
            return " > ".join(str(x) for x in v)
        return v

    with (OUT_DIR / "top20_detail.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for row in rows:
            w.writerow([flat(row.get(k)) for k in CSV_FIELDS])


# ---------------------------------------------------------------------------
# item 2 -- error_examples.md
# ---------------------------------------------------------------------------

def load_error_csv() -> list:
    path = ROOT / "results/bottleneck_root_cause/top1_error_detail.csv"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_example(row: dict, records: dict, tabs: dict) -> list:
    qid = row["query_id"]
    q = records[qid].get("question", "")
    gtext = serialize_cell(row["gold_table"], int(row["gold_row"]), int(row["gold_col"]), tabs)
    ttext = serialize_cell(row["top1_table"], int(row["top1_row"]), int(row["top1_col"]), tabs)
    return [
        f"### {qid}", "",
        f"- query: {q}",
        f"- gold: table={row['gold_table']} ({row['gold_row']},{row['gold_col']}) "
        f"path=`{row['gold_row_path']} | {row['gold_col_path']}` "
        f"serialized: {gtext}",
        f"- top1: table={row['top1_table']} ({row['top1_row']},{row['top1_col']}) "
        f"path=`{row['top1_row_path']} | {row['top1_col_path']}` "
        f"serialized: {ttext}",
        f"- dense_score gold={row['dense_score_gold']} top1={row['dense_score_top1']} "
        f"margin={row['margin_dense']}",
        f"- hybrid_score gold={row['hybrid_score_gold']} top1={row['hybrid_score_top1']} "
        f"margin={row['margin_hybrid']}",
        f"- gold_rank={row['gold_rank']} error_class={row['error_class']}", "",
    ]


def rank_gt20_unknown_examples(pop: dict, gold_path_rows: dict, tabs: dict) -> list:
    """Every query whose gold_rank (RECORDS scan) is >20 or unknown (None) --
    ``top1`` here is that query's OWN context_units[0], since these queries have
    no gold_path top1 to lean on for anything but the score (>20 or unknown means
    the gold cell itself was never top-1)."""
    out = []
    for qid, r in sorted(pop.items()):
        gr = r.get("gold_rank")
        if gr is not None and gr <= 20:
            continue
        gtid, gi, gj = sorted(r["gold_cells"])[0]
        units = r.get("context_units") or []
        ttid, ti, tj = (tuple(units[0]["cells"][0]) if units else (None, None, None))
        gtext = serialize_cell(gtid, gi, gj, tabs)
        ttext = serialize_cell(ttid, ti, tj, tabs) if ttid else None
        gp = gold_path_rows.get(qid)
        out.append({
            "query_id": qid, "status": "unknown" if gr is None else f"rank={gr} (>20)",
            "query": r.get("question"), "gold_table": gtid, "gold_cell": [gi, gj],
            "gold_serialized_text": gtext, "gold_table_in_context": r.get("gold_table_in_context"),
            "top1_table": ttid, "top1_cell": [ti, tj] if ttid else None,
            "top1_serialized_text": ttext,
            "dense_score_gold": gp["dense_score_gold"] if gp else None,
            "dense_score_top1": gp["dense_score_top1"] if gp else None,
            "hybrid_score_gold": gp["hybrid_score_gold"] if gp else None,
            "hybrid_score_top1": gp["hybrid_score_top1"] if gp else None,
        })
    return out


def write_error_examples(records: dict, tabs: dict) -> dict:
    pop = load_primary_population()
    err_rows = load_error_csv()
    by_class = defaultdict(list)
    success = []
    for row in err_rows:
        if row["error_class"] == "exact_match":
            success.append(row)
        else:
            by_class[row["error_class"]].append(row)

    lines = ["# 오류 사례 (사전 개선 진단)", "",
             "모집단: hitab test primary (mode=all, m=1, aggregation=none), query count=991 "
             "(results/bottleneck_root_cause/top1_error_detail.csv 그대로 사용, "
             "새로 채점하지 않음). 예시는 query_id 오름차순 처음 20건.", ""]

    counts = {}
    for label, pool in (("top1 성공", success), ("wrong_table", by_class["wrong_table"]),
                        ("wrong_row", by_class["wrong_row"]), ("wrong_column", by_class["wrong_column"])):
        pool_sorted = sorted(pool, key=lambda r: r["query_id"])
        picked = pool_sorted[:20]
        counts[label] = {"n_total_in_class": len(pool_sorted), "n_shown": len(picked)}
        lines += [f"## {label} (전체 {len(pool_sorted)}건 중 {len(picked)}건)", ""]
        for row in picked:
            lines += render_example(row, records, tabs)

    gold_path_rows = load_retrieval_rows("gold_path")
    gt20 = rank_gt20_unknown_examples(pop, gold_path_rows, tabs)
    n_unknown = sum(1 for x in gt20 if x["status"] == "unknown")
    n_gt20 = len(gt20) - n_unknown
    lines += [f"## gold_rank > 20 (전체 {n_gt20}건, 전부 표시)", ""]
    for x in gt20:
        if x["status"] == "unknown":
            continue
        lines += [
            f"### {x['query_id']} ({x['status']})", "",
            f"- query: {x['query']}",
            f"- gold: table={x['gold_table']} {x['gold_cell']} "
            f"gold_table_in_context={x['gold_table_in_context']} "
            f"serialized: {x['gold_serialized_text']}",
            f"- top1(this query's own): table={x['top1_table']} {x['top1_cell']} "
            f"serialized: {x['top1_serialized_text']}",
            f"- dense_score gold={x['dense_score_gold']} top1={x['dense_score_top1']}",
            f"- hybrid_score gold={x['hybrid_score_gold']} top1={x['hybrid_score_top1']}", "",
        ]
    lines += [f"## gold_rank unknown (top-500 스캔 밖, 전체 {n_unknown}건, 전부 표시)", ""]
    for x in gt20:
        if x["status"] != "unknown":
            continue
        lines += [
            f"### {x['query_id']} (unknown)", "",
            f"- query: {x['query']}",
            f"- gold: table={x['gold_table']} {x['gold_cell']} "
            f"gold_table_in_context={x['gold_table_in_context']} "
            f"serialized: {x['gold_serialized_text']}",
            f"- top1(this query's own): table={x['top1_table']} {x['top1_cell']} "
            f"serialized: {x['top1_serialized_text']}",
            f"- dense_score gold={x['dense_score_gold']} top1={x['dense_score_top1']}",
            f"- hybrid_score gold={x['hybrid_score_gold']} top1={x['hybrid_score_top1']}", "",
        ]
    (OUT_DIR / "error_examples.md").write_text("\n".join(lines), encoding="utf-8")
    return {"counts": counts, "n_rank_gt20": n_gt20, "n_unknown": n_unknown}


# ---------------------------------------------------------------------------
# item 3 -- retrieval oracle diagnosis (diagnostic-only gold filters)
# ---------------------------------------------------------------------------

def compat(cand, gold, mode, tabs) -> bool:
    ctid, ci, cj = cand
    gtid, gi, gj = gold
    if mode == "table":
        return ctid == gtid
    if mode == "table_row":
        return ctid == gtid and ci == gi
    if mode == "table_col":
        return ctid == gtid and cj == gj
    crp, ccp, _ = cell_paths(ctid, ci, cj, tabs)
    grp, gcp, _ = cell_paths(gtid, gi, gj, tabs)
    if mode == "row":
        return crp is not None and crp == grp
    if mode == "col":
        return ccp is not None and ccp == gcp
    raise ValueError(mode)


ORACLE_CONDITIONS = [
    ("global", None), ("gold_table_only", "table"),
    ("gold_row_compatible", "row"), ("gold_column_compatible", "col"),
    ("gold_table_and_row", "table_row"), ("gold_table_and_column", "table_col"),
]


def oracle_metrics_for(mode, pop: dict, tabs: dict) -> dict:
    if mode is None:
        ranks = [r["gold_rank"] for r in pop.values() if r.get("gold_rank") is not None]
        n_unknown = sum(1 for r in pop.values() if r.get("gold_rank") is None)
        n = len(pop)
        return {
            "population_definition": "primary single-cell population (m=1, mode=all, "
                                     "aggregation=none), gold_rank from the deployed "
                                     "top-500-capped scan (no filter -- the real method)",
            "n_population": n, "n_known_rank": len(ranks), "n_unknown_rank": n_unknown,
            "recall_at_1": round(sum(x == 1 for x in ranks) / n, 4),
            "recall_at_5": round(sum(x <= 5 for x in ranks) / n, 4),
            "recall_at_20": round(sum(x <= 20 for x in ranks) / n, 4),
            "mrr_over_known_rank_only": round(sum(1 / x for x in ranks) / len(ranks), 4),
            "esm_at_1": round(sum(x == 1 for x in ranks) / n, 4),
            "esm_note": "single gold cell -> ESM>0 only possible at rank 1, equal to recall_at_1",
        }
    observable, n_censored = [], 0
    for qid, r in pop.items():
        gr = r.get("gold_rank")
        units = r.get("context_units") or []
        if gr is None or gr > 20 or not units:
            n_censored += 1
            continue
        gold = tuple(sorted(r["gold_cells"])[0])
        cand_cells = [tuple(u["cells"][0]) for u in units[:gr]]
        oracle_rank = sum(1 for c in cand_cells if c != gold and compat(c, gold, mode, tabs)) + 1
        observable.append(oracle_rank)
    n = len(observable)
    return {
        "population_definition": "DIAGNOSTIC ORACLE, not a real retrieval method: gold "
                                 "table/row/column identity is used as a filter. Computed "
                                 "only over queries observable in the existing top-20 "
                                 "window (gold_rank<=20); relative order within that window "
                                 "is preserved from the already-ranked context_units, so no "
                                 "rescoring/model call is needed. Queries with gold_rank>20 "
                                 "or unknown cannot be resolved from the top-20 window alone "
                                 "and are reported as censored, not estimated.",
        "n_observable": n, "n_censored_gold_rank_gt20_or_unknown": n_censored,
        "recall_at_1": round(sum(x == 1 for x in observable) / n, 4) if n else None,
        "recall_at_5": round(sum(x <= 5 for x in observable) / n, 4) if n else None,
        "recall_at_20": round(sum(x <= 20 for x in observable) / n, 4) if n else None,
        "mrr": round(sum(1 / x for x in observable) / n, 4) if n else None,
        "esm_at_1": round(sum(x == 1 for x in observable) / n, 4) if n else None,
        "esm_note": "single gold cell -> ESM>0 only possible at rank 1, equal to recall_at_1",
    }


def write_oracle_diagnosis(tabs: dict) -> dict:
    pop = load_primary_population()
    out = {"note": "gold-table/row/column filters below are diagnostic oracles used to "
                   "bound where retrieval loses the gold cell; they are never used to "
                   "select or score anything and are not a proposed retrieval method.",
           "conditions": {}}
    for name, mode in ORACLE_CONDITIONS:
        out["conditions"][name] = oracle_metrics_for(mode, pop, tabs)
    (OUT_DIR / "oracle_diagnosis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# item 4 -- integrity audit
# ---------------------------------------------------------------------------

def stats(xs):
    xs = list(xs)
    if not xs:
        return None
    return {"n": len(xs), "min": min(xs), "max": max(xs),
            "mean": round(mean(xs), 2), "median": median(xs)}


def representation_integrity(representation: str) -> dict:
    texts, coords, fallback = build_repr_corpus(representation, DATA_DIR)
    coord_index = {}
    dup_coords = 0
    for c in coords:
        if c in coord_index:
            dup_coords += 1
        else:
            coord_index[c] = True
    text_counts = Counter(texts)
    collisions = sum(1 for c, n in text_counts.items() if n > 1)
    collision_cells = sum(n for n in text_counts.values() if n > 1)
    # encoder metadata is read straight off the already-cached embedding file's
    # sibling summary (results/bottleneck_root_cause/retrieval_<repr>.json) --
    # no encoder is constructed here, so no model is touched.
    summary_path = ROOT / f"results/bottleneck_root_cause/retrieval_{representation}.json"
    encoder_name = None
    cache_hit = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        encoder_name = s.get("encoder")
        cache_hit = s.get("embed_cache_hit")
    return {
        "representation": representation, "n_units": len(texts),
        "n_predicted_path_fallback_to_value_only": fallback if representation == "predicted_path" else None,
        "n_duplicate_coordinates": dup_coords,
        "n_distinct_serialized_texts_with_collision": collisions,
        "n_cells_involved_in_a_text_collision": collision_cells,
        "encoder_name_from_prior_run": encoder_name,
        "prior_run_embed_cache_hit": cache_hit,
        "text_length_chars": stats(len(t) for t in texts),
    }


def cache_key_collisions(reps: dict) -> dict:
    """Whether the three representations' texts could collide on the same
    cache filename. No encoder is constructed (that would load the model
    just to read its name) -- ``encode_corpus``'s key is
    ``digest({"texts": texts, "encoder": encoder.metadata(), ...})``, and the
    already-recorded ``retrieval_<repr>.json`` summaries (read in
    representation_integrity) show each of the three runs got
    ``embed_cache_hit`` consistent with its OWN ``n_units`` -- a real
    collision would have loaded a wrong-shaped array and raised there
    already. This only adds the text-level check that is actually new:
    distinct representations produce distinct texts (so distinct digests,
    SHA256 collision being cryptographically negligible)."""
    text_sets = {rep: tuple(build_repr_corpus(rep, DATA_DIR)[0]) for rep in REPRESENTATIONS}
    all_identical_pairs = [(a, b) for i, a in enumerate(REPRESENTATIONS)
                           for b in REPRESENTATIONS[i + 1:] if text_sets[a] == text_sets[b]]
    return {"n_representations": len(REPRESENTATIONS),
            "representations_with_identical_text_sets": all_identical_pairs,
            "prior_runs_embed_cache_hit": {rep: reps[rep]["prior_run_embed_cache_hit"] for rep in reps},
            "note": "no encoder constructed/loaded for this check; see note above"}


def unknown_rank_causes(tabs: dict) -> list:
    gold_path_coords = set(build_repr_corpus("gold_path", DATA_DIR)[1])
    pop = load_primary_population()
    out = []
    for qid in unknown_rank_query_ids():
        r = pop[qid]
        gtid, gi, gj = sorted(r["gold_cells"])[0]
        indexed = (gtid, gi, gj) in gold_path_coords
        rp, cp, val = cell_paths(gtid, gi, gj, tabs)
        out.append({
            "query_id": qid, "gold_table": gtid, "gold_cell": [gi, gj],
            "gold_value": val, "indexed_in_split_corpus": indexed,
            "gold_table_in_context": r.get("gold_table_in_context"),
            "cause": ("gold cell IS indexed but scored outside the top-500 "
                     "hybrid-ranked units -- a genuine retrieval miss, not a "
                     "corpus/indexing gap"
                     if indexed and not r.get("gold_table_in_context") else
                     "gold cell IS indexed and its table DOES appear in the "
                     "delivered top-20 via a different cell, but the exact gold "
                     "coordinate itself still scored outside the top-500 -- a "
                     "cell-level retrieval miss, not a table-level one"
                     if indexed else
                     "gold cell coordinate is NOT present in the split corpus's "
                     "indexed units -- a corpus/indexing gap, not a ranking miss"),
        })
    return out


def write_integrity_audit(records: dict, tabs: dict) -> None:
    scored = {qid: r for qid, r in records.items() if "correct" in r}
    excluded = {qid: r for qid, r in records.items() if "correct" not in r}
    tids_all = sorted({r["table_id"] for r in records.values() if "table_id" in r})
    split_tids = split_corpus_table_ids()
    m_dist = Counter(r["m"] for r in scored.values())
    mode_agg_dist = Counter((r["mode"], r.get("aggregation")) for r in scored.values())

    reps = {rep: representation_integrity(rep) for rep in REPRESENTATIONS}
    cache_check = cache_key_collisions(reps)
    causes = unknown_rank_causes(tabs)

    # table size / path length distributions over the split corpus (538 tables)
    row_counts, col_counts, row_path_lens, col_path_lens = [], [], [], []
    for tid in split_tids:
        tab = get_table(tid, tabs)
        if tab is None:
            continue
        t = tab.table
        row_counts.append(t.n_rows)
        col_counts.append(t.n_cols)
        for i in range(t.n_rows):
            row_path_lens.append(len(t.row_path(i)))
        for j in range(t.n_cols):
            col_path_lens.append(len(t.col_path(j)))

    n_scored_gold_indexed_missing = sum(
        1 for qid, r in scored.items()
        if r["mode"] == "all" and r["m"] == 1
        and tuple(sorted(r["gold_cells"])[0]) not in
        set(build_repr_corpus("gold_path", DATA_DIR)[1]))

    lines = ["# 무결성 통계 (사전 개선 진단)", "",
             "## query / table / indexed-cell 수", "",
             f"- 전체 query: {len(records)} (채점 {len(scored)}, 제외 {len(excluded)})",
             f"- 제외 사유: {dict(Counter(r['excluded'] for r in excluded.values()))}",
             f"- query가 참조하는 distinct table: {len(tids_all)}",
             f"- split corpus가 색인하는 table: {len(split_tids)}",
             ""]
    for rep, d in reps.items():
        lines.append(f"- {rep} 색인 단위 수: {d['n_units']} "
                     f"(predicted_path fallback to value_only: {d['n_predicted_path_fallback_to_value_only']})")
    lines += ["", "## gold 개수(m) 분포", "", "| m | query count |", "|---|---|"]
    for m, n in sorted(m_dist.items()):
        lines.append(f"| {m} | {n} |")
    lines += ["", "## (mode, aggregation) 분포", "", "| mode | aggregation | query count |", "|---|---|---|"]
    for (mode, agg), n in sorted(mode_agg_dist.items(), key=lambda x: -x[1]):
        lines.append(f"| {mode} | {agg} | {n} |")

    lines += ["", "## duplicate cell_id / 직렬화 텍스트 충돌 (representation별)", "",
             "| representation | n_units | duplicate coords | text-collision distinct-texts | "
             "text-collision 관련 cell 수 |", "|---|---|---|---|---|"]
    for rep, d in reps.items():
        lines.append(f"| {rep} | {d['n_units']} | {d['n_duplicate_coordinates']} | "
                     f"{d['n_distinct_serialized_texts_with_collision']} | "
                     f"{d['n_cells_involved_in_a_text_collision']} |")

    lines += ["", "## representation별 embedding cache key 충돌 여부", "",
             f"- 확인한 representation 수: {cache_check['n_representations']}",
             f"- 텍스트 집합이 동일한 representation 쌍(있으면 잠재적 충돌): "
             f"{cache_check['representations_with_identical_text_sets']}",
             f"- 각 representation의 사전 실행 embed_cache_hit: "
             f"{cache_check['prior_runs_embed_cache_hit']}", ""]

    lines += ["", "## index에 없는 gold (primary population, m=1)", "",
             f"- gold_path corpus에 색인되지 않은 gold cell 수: {n_scored_gold_indexed_missing} / "
             f"{sum(1 for r in scored.values() if r['mode']=='all' and r['m']==1)}", ""]

    lines += ["## unknown-rank 6건 원인", ""]
    for c in causes:
        lines.append(f"- {c['query_id']} (table={c['gold_table']}, cell={c['gold_cell']}, "
                     f"value={c['gold_value']!r}, indexed={c['indexed_in_split_corpus']}, "
                     f"gold_table_in_context={c['gold_table_in_context']}): {c['cause']}")

    lines += ["", "## table 크기 / path 길이 / 직렬화 길이 분포 (split corpus, 538 tables)", "",
             f"- n_rows per table: {stats(row_counts)}",
             f"- n_cols per table: {stats(col_counts)}",
             f"- row_path 길이(segment 수): {stats(row_path_lens)}",
             f"- col_path 길이(segment 수): {stats(col_path_lens)}", ""]
    for rep, d in reps.items():
        lines.append(f"- {rep} 직렬화 텍스트 길이(chars): {d['text_length_chars']}")

    (OUT_DIR / "integrity_audit.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# top-level report
# ---------------------------------------------------------------------------

def write_top_md(n_rows: int, n_queries: int, err_summary: dict, oracle: dict) -> None:
    g = oracle["conditions"]["global"]
    lines = ["# 사전 개선 진단 (PRE_IMPROVEMENT_AUDIT)", "",
             "모델/랭킹 미변경, 신규 API 호출 없음. 기존 retrieval 결과("
             "results/evaluation_v2/s3c_v2_records.jsonl, "
             "results/bottleneck_root_cause/*)와 data/hitab 테이블 파일만 사용.", "",
             f"- 전체 query {n_queries}건, top20_detail 행 {n_rows}건",
             f"- global: R@1={g['recall_at_1']} R@5={g['recall_at_5']} R@20={g['recall_at_20']} "
             f"MRR={g['mrr_over_known_rank_only']} (query count={g['n_population']}, unknown_rank={g['n_unknown_rank']})",
             f"- top1 성공 {err_summary['counts']['top1 성공']['n_total_in_class']}건, "
             f"wrong_table {err_summary['counts']['wrong_table']['n_total_in_class']}건, "
             f"wrong_row {err_summary['counts']['wrong_row']['n_total_in_class']}건, "
             f"wrong_column {err_summary['counts']['wrong_column']['n_total_in_class']}건",
             f"- gold_rank>20 {err_summary['n_rank_gt20']}건, unknown {err_summary['n_unknown']}건",
             "", "## 생성 파일", "",
             "- top20_detail.jsonl / top20_detail.csv",
             "- error_examples.md", "- oracle_diagnosis.json", "- integrity_audit.md", ""]
    (OUT_DIR / "PRE_IMPROVEMENT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all"])
    ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabs: dict = {}
    records = read_records(RECORDS_PATH)
    gold_path_rows = load_retrieval_rows("gold_path")

    rows = build_top20_rows(records, gold_path_rows, tabs)
    write_top20(rows)
    print(f"top20_detail: {len(rows)} rows / {len(records)} queries")

    err_summary = write_error_examples(records, tabs)
    print(f"error_examples: {err_summary}")

    oracle = write_oracle_diagnosis(tabs)
    print("oracle_diagnosis conditions:", list(oracle["conditions"]))

    write_integrity_audit(records, tabs)
    print("integrity_audit.md written")

    write_top_md(len(rows), len(records), err_summary, oracle)
    print("PRE_IMPROVEMENT_AUDIT.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
