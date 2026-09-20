#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Root-cause diagnosis for the k-ladder bottleneck (BOTTLENECK_DIAGNOSIS.md),
testing three hypotheses with the SAME primary population, encoder, hybrid
formula (alpha=0.7, untouched — no test-split tuning) and n=150 controlled-QA
sample already used there. Reuses scripts/bottleneck_diagnosis.py's population
loader, value-index builder, top-1 error classifier and controlled-QA distractor
pickers; reuses scripts/retrieval_accuracy.py's cell_unit renderer and embedding
cache (same cache key formula -> the gold_path representation below is
byte-identical text to the deployed s3c corpus and should cache-hit); reuses
scripts/tree_reconstruct_hitab_raw.py's path-comparison metrics (norm,
segment_f1, hierarchy_edges) for structure scoring.

  H1  predicted 구조 복원 오류      — is the RECONSTRUCTED header path (from the
                                      raw grid alone, rag_agent.reconstruct.
                                      header_grid) wrong vs gold?
  H2  구조는 맞지만 embedding이 무시 — does the embedding actually reward a cell
                                      sentence carrying header structure over
                                      one carrying only the bare value?
  H3  구조적 hard-negative가 방해    — does a structurally-plausible decoy (same
                                      table, wrong row/column) hurt the READER
                                      more than a random or unrelated one?

Predicted structure comes from rag_agent.reconstruct.header_grid.
reconstruct_{row,col}_paths on the RAW grid, mapped to data cells via
hitab_grid.HitabTable.row_map/col_map (the pipeline's own exact raw<->data
map — NOT tree_reconstruct_hitab_raw.py's value-matching align(), which exists
only to locate data lines in a grid that carries no separate parse; hitab_grid
already has the exact map). The header BLOCK SIZE (nhr/nhc) stays gold
(min of the mapped lines) — only the path CONTENT within it is reconstructed,
matching tree_reconstruct_hitab_raw.py's default (non-guessed-boundary) mode.
This recovery is non-destructive (nothing deleted or restored from git); if it
had failed the required fallback is to record the missing fields and skip H1,
per instruction.

ESM keeps its existing definition (rag_agent unchanged) and its existing
structural fact: with one gold cell and a fixed k, ESM>0 only at k=1. Gold
cells are used for scoring and the gold-only/oracle controlled-QA condition
ONLY — never to select or rank anything. alpha=0.7 and every other retrieval
hyperparameter are the values already fixed elsewhere in this repo, not
re-picked here (no test-split tuning). The 6 queries whose gold_rank was
``None`` under retrieval_accuracy.py's capped (top-500) scan are called out
separately everywhere they matter, not folded into ">20".

  PYTHONPATH=. .venv/bin/python scripts/bottleneck_root_cause.py structure
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_root_cause.py retrieval --representation value_only
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_root_cause.py retrieval --representation gold_path
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_root_cause.py retrieval --representation predicted_path
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_root_cause.py errorcsv
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_root_cause.py hardneg
  PYTHONPATH=. .venv/bin/python scripts/bottleneck_root_cause.py verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.artifacts import digest, read_records             # noqa: E402
from rag_agent.reconstruct import reconstruct_col_paths, reconstruct_row_paths  # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize        # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import with_page_title           # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES, cell_unit         # noqa: E402
from scripts.tree_reconstruct_hitab_raw import (hierarchy_edges, norm,  # noqa: E402
                                                segment_f1)
from bottleneck_diagnosis import (classify_top1_error,                # noqa: E402
                                  load_primary_population, pick_hard_negative,
                                  pick_random, split_corpus_table_ids)

OUT_DIR = ROOT / "results/bottleneck_root_cause"
BD_DIR = ROOT / "results/bottleneck_diagnosis"
EMBED_CACHE = ROOT / ".cache/retrieval_accuracy"
ALPHA = 0.7                       # fixed elsewhere in this repo; not re-picked here
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}
REPRESENTATIONS = ("value_only", "gold_path", "predicted_path")


def unknown_rank_query_ids() -> list:
    """The queries whose gold_rank was None under retrieval_accuracy.py's
    top-500-capped scan (s3c_v2_records.jsonl) — reported separately, per
    instruction, not folded into a rank bucket."""
    pop = load_primary_population()
    return sorted(qid for qid, r in pop.items() if r.get("gold_rank") is None)


# ---------------------------------------------------------------------------
# predicted structure (H1)
# ---------------------------------------------------------------------------

_STRUCT_CACHE: dict = {}


def predicted_structure(tid: str, data_dir: str = "data/hitab"):
    """``(row_path_fn, col_path_fn, meta)``; ``(None, None, reason)`` if the raw
    grid or the exact raw<->data map (hitab_grid) is unusable for this table —
    recorded and skipped, never guessed."""
    if tid in _STRUCT_CACHE:
        return _STRUCT_CACHE[tid]
    tab = hg.load_table(tid, data_dir)
    if tab is None or not tab.row_map or not tab.col_map:
        out = (None, None, "no_raw_or_unmapped_axis")
        _STRUCT_CACHE[tid] = out
        return out
    texts = tab.raw.get("texts") or []
    nhr, nhc = min(tab.row_map), min(tab.col_map)
    if nhr <= 0 or nhc <= 0 or not texts:
        out = (None, None, "degenerate_header_block")
        _STRUCT_CACHE[tid] = out
        return out
    rec_rows = reconstruct_row_paths(texts, nhr, nhc)
    rec_cols = reconstruct_col_paths(texts, nhr, nhc)
    inv_row = {v: k for k, v in tab.row_map.items()}
    inv_col = {v: k for k, v in tab.col_map.items()}

    def row_path(i):
        raw = inv_row.get(i)
        if raw is None:
            return None
        idx = raw - nhr
        return rec_rows[idx] if 0 <= idx < len(rec_rows) else None

    def col_path(j):
        raw = inv_col.get(j)
        if raw is None:
            return None
        idx = raw - nhc
        return rec_cols[idx] if 0 <= idx < len(rec_cols) else None

    out = (row_path, col_path, {"nhr": nhr, "nhc": nhc})
    _STRUCT_CACHE[tid] = out
    return out


def structure_metrics(table_ids, data_dir: str = "data/hitab") -> dict:
    """Aggregate row/col path exact match, segment F1 (path-node F1), and
    parent-child edge P/R/F1 — reconstructed vs GOLD path, over every data
    cell of ``table_ids``. Same definitions as
    scripts/tree_reconstruct_hitab_raw.py (imported, not re-derived)."""
    col_hit = col_tot = row_hit = row_tot = 0
    col_f1: list = []
    row_f1: list = []
    pairs = {"col": [0, 0, 0], "row": [0, 0, 0]}
    excluded = Counter()
    for tid in table_ids:
        row_path_fn, col_path_fn, meta = predicted_structure(tid, data_dir)
        if row_path_fn is None:
            excluded[meta] += 1
            continue
        t = hg.load_table(tid, data_dir).table
        gold_rows, rec_rows_ = [], []
        for i in range(t.n_rows):
            rp = row_path_fn(i)
            if rp is None:
                continue
            gp = t.row_path(i)
            row_tot += 1
            ok = norm(gp) == norm(rp)
            row_hit += int(ok)
            row_f1.append(segment_f1(gp, rp))
            gold_rows.append(gp); rec_rows_.append(rp)
        gold_cols, rec_cols_ = [], []
        for j in range(t.n_cols):
            cp = col_path_fn(j)
            if cp is None:
                continue
            gp = t.col_path(j)
            col_tot += 1
            ok = norm(gp) == norm(cp)
            col_hit += int(ok)
            col_f1.append(segment_f1(gp, cp))
            gold_cols.append(gp); rec_cols_.append(cp)
        for axis, gp_list, rp_list in (("row", gold_rows, rec_rows_), ("col", gold_cols, rec_cols_)):
            g, rc = hierarchy_edges(gp_list), hierarchy_edges(rp_list)
            pairs[axis][0] += len(g & rc); pairs[axis][1] += len(rc); pairs[axis][2] += len(g)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    def prf(hit, n_rec, n_gold):
        p = hit / n_rec if n_rec else 0.0
        r = hit / n_gold if n_gold else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
                "n_gold_edges": n_gold, "n_rec_edges": n_rec}

    return {"n_tables_total": len(table_ids),
           "n_tables_excluded": sum(excluded.values()),
           "excluded_reasons": dict(excluded),
           "row_path_exact_match": round(row_hit / row_tot, 4) if row_tot else None,
           "col_path_exact_match": round(col_hit / col_tot, 4) if col_tot else None,
           "row_path_segment_f1_path_node_f1": mean(row_f1),
           "col_path_segment_f1_path_node_f1": mean(col_f1),
           "row_pair_prf": prf(*pairs["row"]), "col_pair_prf": prf(*pairs["col"]),
           "row_paths_scored": row_tot, "col_paths_scored": col_tot}


def cell_structure_correct(tid, i, j, data_dir="data/hitab"):
    """``True``/``False``/``None`` (undetermined — structure unavailable)."""
    row_path_fn, col_path_fn, _meta = predicted_structure(tid, data_dir)
    if row_path_fn is None:
        return None
    t = hg.load_table(tid, data_dir).table
    rp, cp = row_path_fn(i), col_path_fn(j)
    if rp is None or cp is None:
        return None
    return norm(rp) == norm(t.row_path(i)) and norm(cp) == norm(t.col_path(j))


def structure_population_joint(data_dir: str = "data/hitab") -> dict:
    """Per primary-population query: whether ITS gold cell's row+col path was
    reconstructed correctly (the join structure-correct/wrong is decomposed
    against in `retrieval` and `errorcsv`)."""
    pop = load_primary_population()
    out = {}
    for qid, r in pop.items():
        tid, i, j = sorted(r["gold_cells"])[0]
        out[qid] = cell_structure_correct(tid, i, j, data_dir)
    return out


# ---------------------------------------------------------------------------
# three-representation corpus + retrieval (H1, H2)
# ---------------------------------------------------------------------------

def build_repr_corpus(representation: str, data_dir: str = "data/hitab"):
    """``(texts, coords, n_predicted_path_fallback)``. ``coords[i]`` is the
    ``(table_id, row, col)`` unit ``texts[i]`` renders. ``predicted_path``
    falls back to a bare value (never to gold) for the cells of a table whose
    structure could not be reconstructed (see predicted_structure) — counted,
    never silent."""
    if representation not in REPRESENTATIONS:
        raise ValueError(representation)
    texts, coords = [], []
    fallback = 0
    for tid in split_corpus_table_ids():
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
        row_path_fn = col_path_fn = None
        if representation == "predicted_path":
            row_path_fn, col_path_fn, _meta = predicted_structure(tid, data_dir)
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = t.data[i][j]
                if not str(v).strip():
                    continue
                if representation == "value_only":
                    txt = str(v)
                elif representation == "gold_path":
                    txt = cell_unit(title, t.row_path(i), t.col_path(j), v, "s3c")
                else:
                    rp = row_path_fn(i) if row_path_fn else None
                    cp = col_path_fn(j) if col_path_fn else None
                    if rp is None or cp is None:
                        txt = str(v)
                        fallback += 1
                    else:
                        txt = cell_unit(title, rp, cp, v, "s3c")
                texts.append(txt)
                coords.append((tid, i, j))
    return texts, coords, fallback


def encode_corpus(texts, encoder):
    """Same cache key formula as scripts/retrieval_accuracy.py, same
    directory — identical texts (gold_path IS the deployed s3c corpus)
    reuse that run's embeddings instead of re-encoding."""
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    key = digest({"texts": texts, "encoder": encoder.metadata(), "overflow": "error"})[:24]
    f = EMBED_CACHE / f"{encoder.name.replace('/', '_')}_{len(texts)}_{key}.npy"
    if f.exists():
        emb = np.load(f)
        return emb, True
    emb = encoder.encode(texts)
    if emb.ndim != 2 or len(emb) != len(texts) or not np.isfinite(emb).all():
        raise ValueError("encoder produced invalid document vectors")
    np.save(f, emb)
    return emb, False


def run_retrieval(representation: str, data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    texts, coords, fallback = build_repr_corpus(representation, data_dir)
    coord_index = {c: idx for idx, c in enumerate(coords)}
    if len(coord_index) != len(coords):
        raise ValueError("cell-unit corpus has a duplicate coordinate")

    enc = default_encoder()
    emb, cache_hit = encode_corpus(texts, enc)
    bm = SparseBM25(_tokenize(t) for t in texts)

    rows = []
    for qid, r in sorted(pop.items()):
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            rows.append({"query_id": qid, "excluded": "gold_cell_not_in_corpus"})
            continue
        q = r["question"]
        dense = emb @ enc.encode_query([q])[0].astype(np.float32)
        sparse = bm.get_scores(_tokenize(q))
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
        order = np.argsort(-hybrid, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        gold_rank = int(ranks[gidx])
        top1 = int(order[0])
        top1_coord = coords[top1]
        rows.append({
            "query_id": qid, "gold_cell": list(gold), "top1_cell": list(top1_coord),
            "gold_rank": gold_rank, "esm": int(gold_rank == 1 and top1_coord == gold),
            "dense_score_gold": float(dense[gidx]), "dense_score_top1": float(dense[top1]),
            "sparse_score_gold": float(sparse[gidx]), "sparse_score_top1": float(sparse[top1]),
            "hybrid_score_gold": float(hybrid[gidx]), "hybrid_score_top1": float(hybrid[top1]),
            "margin_hybrid_top1_minus_gold": float(hybrid[top1] - hybrid[gidx]),
            "margin_dense_top1_minus_gold": float(dense[top1] - dense[gidx]),
        })

    scored = [r for r in rows if "excluded" not in r]
    ranks_arr = [r["gold_rank"] for r in scored]

    def recall_at(k):
        return round(sum(x <= k for x in ranks_arr) / len(ranks_arr), 4) if ranks_arr else None

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else None

    summary = {
        "representation": representation, "alpha": ALPHA, "encoder": enc.name,
        "embed_cache_hit": cache_hit, "n_units": len(texts),
        "n_predicted_path_fallback_to_value_only": fallback if representation == "predicted_path" else None,
        "n_population": len(pop), "n_scored": len(scored),
        "n_excluded": len(rows) - len(scored),
        "recall_at_1": recall_at(1), "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10), "recall_at_20": recall_at(20),
        "esm_at_1_note": "ESM defined as before: >0 only possible at k=1 (single gold cell)",
        "mrr": round(sum(1.0 / x for x in ranks_arr) / len(ranks_arr), 4) if ranks_arr else None,
        "median_rank": median(ranks_arr),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"retrieval_{representation}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT_DIR / f"retrieval_{representation}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def load_retrieval_rows(representation: str) -> dict:
    path = OUT_DIR / f"retrieval_{representation}.jsonl"
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            out[row["query_id"]] = row
    return out


def decompose_by_structure(data_dir: str = "data/hitab") -> dict:
    """retrieval (predicted_path representation) split by whether the gold
    cell's OWN structure reconstruction was correct — the direct H1 test —
    plus, for the structure-correct subset, the paired gold_path-vs-
    predicted_path comparison (the residual gap, if any, once reconstruction
    itself is right)."""
    joint = structure_population_joint(data_dir)
    pred_rows = load_retrieval_rows("predicted_path")
    gold_rows = load_retrieval_rows("gold_path")

    def block(qids, rowset):
        ranks = [rowset[q]["gold_rank"] for q in qids if q in rowset and "excluded" not in rowset[q]]
        if not ranks:
            return {"n": 0}
        return {"n": len(ranks),
               "recall_at_1": round(sum(x <= 1 for x in ranks) / len(ranks), 4),
               "recall_at_5": round(sum(x <= 5 for x in ranks) / len(ranks), 4),
               "recall_at_10": round(sum(x <= 10 for x in ranks) / len(ranks), 4),
               "recall_at_20": round(sum(x <= 20 for x in ranks) / len(ranks), 4),
               "mrr": round(sum(1.0 / x for x in ranks) / len(ranks), 4),
               "median_rank": sorted(ranks)[len(ranks) // 2]}

    correct = [q for q, v in joint.items() if v is True]
    wrong = [q for q, v in joint.items() if v is False]
    undetermined = [q for q, v in joint.items() if v is None]
    return {"n_structure_correct": len(correct), "n_structure_wrong": len(wrong),
           "n_structure_undetermined": len(undetermined),
           "predicted_path__structure_correct": block(correct, pred_rows),
           "predicted_path__structure_wrong": block(wrong, pred_rows),
           "predicted_path__structure_undetermined": block(undetermined, pred_rows),
           "gold_path__on_structure_correct_subset": block(correct, gold_rows),
           "predicted_path__on_structure_correct_subset": block(correct, pred_rows)}


# ---------------------------------------------------------------------------
# item 2 — top-1 error CSV
# ---------------------------------------------------------------------------

def write_error_csv(data_dir: str = "data/hitab") -> Path:
    pop = load_primary_population()
    gold_rows = load_retrieval_rows("gold_path")
    tabs: dict = {}
    unknown = set(unknown_rank_query_ids())
    out_path = OUT_DIR / "top1_error_detail.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "gold_rank", "unknown_rank_original_scan", "esm",
                   "gold_table", "gold_row", "gold_col", "top1_table", "top1_row", "top1_col",
                   "same_table", "same_row", "same_col", "error_class",
                   "gold_row_path", "gold_col_path", "top1_row_path", "top1_col_path",
                   "dense_score_gold", "dense_score_top1", "margin_dense",
                   "hybrid_score_gold", "hybrid_score_top1", "margin_hybrid"])
        for qid, r in sorted(pop.items()):
            gr = gold_rows.get(qid)
            if gr is None or "excluded" in gr:
                continue
            gtid, gi, gj = gr["gold_cell"]
            ttid, ti, tj = gr["top1_cell"]

            def tab(tid):
                x = tabs.get(tid)
                if x is None:
                    x = tabs[tid] = hg.load_table(tid, data_dir)
                return x

            gt, tt = tab(gtid), tab(ttid)
            cls = ("exact_match" if (gtid, gi, gj) == (ttid, ti, tj)
                  else classify_top1_error((ttid, ti, tj), (gtid, gi, gj), tabs, data_dir))
            w.writerow([qid, gr["gold_rank"], int(qid in unknown), gr["esm"],
                       gtid, gi, gj, ttid, ti, tj,
                       int(gtid == ttid), int(gtid == ttid and gi == ti), int(gtid == ttid and gj == tj),
                       cls,
                       " > ".join(gt.table.row_path(gi)), " > ".join(gt.table.col_path(gj)),
                       " > ".join(tt.table.row_path(ti)), " > ".join(tt.table.col_path(tj)),
                       round(gr["dense_score_gold"], 6), round(gr["dense_score_top1"], 6),
                       round(gr["margin_dense_top1_minus_gold"], 6),
                       round(gr["hybrid_score_gold"], 6), round(gr["hybrid_score_top1"], 6),
                       round(gr["margin_hybrid_top1_minus_gold"], 6)])
    return out_path


# ---------------------------------------------------------------------------
# item 3 — controlled-QA hard-negative reanalysis (NO new reader calls;
# reuses the n=150 legs already run under results/bottleneck_diagnosis/)
# ---------------------------------------------------------------------------

def load_leg(name: str) -> dict:
    path = BD_DIR / f"controlled_qa_{name}.jsonl"
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            out[row["query_id"]] = row
    return out


def margin_bucket(m):
    if m is None:
        return "unknown"
    if m < 0:
        return "distractor_below_gold"
    if m < 0.05:
        return "0.00-0.05"
    if m < 0.15:
        return "0.05-0.15"
    if m < 0.30:
        return "0.15-0.30"
    return ">=0.30"


def hardneg_analysis(data_dir: str = "data/hitab") -> dict:
    pop = load_primary_population()
    gold_rows = load_retrieval_rows("gold_path")
    sample = {qid: pop[qid] for qid in
             json.loads((BD_DIR / "controlled_qa_sample.json").read_text())["query_ids"]}
    baseline = json.loads((BD_DIR / "controlled_qa_gold_only.json").read_text())["answer_accuracy"]
    tabs: dict = {}
    per_leg = {}
    for cond in ("gold_1_random_first", "gold_1_random_last",
                "gold_1_hard_negative_first", "gold_1_hard_negative_last"):
        leg = load_leg(cond)
        dtype = cond.removeprefix("gold_1_").rsplit("_", 1)[0]     # "random" | "hard_negative"
        rows = []
        for qid, r in sample.items():
            if qid not in leg:
                continue                          # dropped at generation time (insufficient candidates)
            gold = tuple(sorted(r["gold_cells"])[0])
            if dtype == "random":
                # reproduce the SAME distractor deterministically (see build_condition_context)
                tab = tabs.get(gold[0]) or hg.load_table(gold[0], data_dir)
                tabs[gold[0]] = tab
                seed_int = int(digest({"cond": cond, "query_id": qid})[:8], 16)
                coords = pick_random(tab, (gold[1], gold[2]), 1, seed_int)
            else:
                coords = pick_hard_negative(r["context_units"], gold, 1)
            if not coords:
                continue
            d = coords[0]
            cls = "exact_match" if d == gold else classify_top1_error(d, gold, tabs, data_dir)
            gr = gold_rows.get(qid)
            margin = None
            if gr is not None and "excluded" not in gr:
                coord_key = tuple(d)
                # margin needs the distractor's OWN hybrid score, which the
                # per-query retrieval dump only kept for gold/top1 — recover it
                # only when the distractor IS the top-1 cell (else unknown, not guessed)
                if coord_key == tuple(gr["top1_cell"]):
                    margin = gr["margin_hybrid_top1_minus_gold"]
            rows.append({"query_id": qid, "distractor_class": cls,
                        "margin_hybrid": margin, "margin_bucket": margin_bucket(margin),
                        "answer_correct": leg[qid]["answer_correct"]})
        by_class = defaultdict(list)
        by_margin = defaultdict(list)
        for row in rows:
            by_class[row["distractor_class"]].append(row["answer_correct"])
            by_margin[row["margin_bucket"]].append(row["answer_correct"])
        per_leg[cond] = {
            "n": len(rows),
            "answer_accuracy": round(sum(r["answer_correct"] for r in rows) / len(rows), 4) if rows else None,
            "by_distractor_class": {c: {"n": len(v), "answer_accuracy": round(sum(v) / len(v), 4)}
                                    for c, v in sorted(by_class.items())},
            "by_margin_bucket_note": "margin known only when the distractor equals the query's own top-1 cell",
            "by_margin_bucket": {b: {"n": len(v), "answer_accuracy": round(sum(v) / len(v), 4)}
                                 for b, v in sorted(by_margin.items())},
        }
    return {"gold_only_baseline_answer_accuracy": baseline, "n_sample": len(sample), "legs": per_leg}


# ---------------------------------------------------------------------------
# verdict + MD
# ---------------------------------------------------------------------------

def verdict(struct: dict, decomp: dict, retr: dict, hardneg: dict) -> dict:
    def v(status, why):
        return {"status": status, "why": why}

    h1 = None
    if struct.get("row_path_exact_match") is not None:
        em = min(struct["row_path_exact_match"], struct["col_path_exact_match"])
        gap = None
        cs, cw = decomp["predicted_path__structure_correct"], decomp["predicted_path__structure_wrong"]
        if cs.get("n") and cw.get("n"):
            gap = round(cs["recall_at_20"] - cw["recall_at_20"], 4)
        if em < 0.7 and gap is not None and gap > 0.1:
            h1 = v("Supported", f"path EM {em} 낮고, structure-correct subset recall@20가 "
                                "wrong subset보다 {gap} 높음")
        elif em < 0.85:
            h1 = v("Partial", f"path EM {em} — 일부 재구성 오류가 존재하나 성공/실패군 recall 차이는 "
                              f"{gap if gap is not None else 'n/a'}")
        else:
            h1 = v("Rejected", f"path EM {em} 로 재구성이 대체로 정확함")
    else:
        h1 = v("Undetermined", "predicted structure를 계산할 표가 없었음")

    h2 = None
    if retr.get("value_only") and retr.get("gold_path"):
        d = retr["gold_path"]["recall_at_20"] - retr["value_only"]["recall_at_20"]
        if d > 0.15:
            h2 = v("Rejected", f"gold_path recall@20 - value_only recall@20 = {round(d,4)} — "
                              "임베딩이 구조를 명확히 반영함")
        elif d > 0.03:
            h2 = v("Partial", f"차이 {round(d,4)} — 구조가 일부만 반영됨")
        else:
            h2 = v("Supported", f"차이 {round(d,4)} — 구조가 사실상 무시됨")
    else:
        h2 = v("Undetermined", "value_only/gold_path retrieval 결과 없음")

    h3 = None
    legs = hardneg.get("legs", {})
    hn = {**legs.get("gold_1_hard_negative_first", {}), }
    rnd = legs.get("gold_1_random_first", {})
    if hn.get("answer_accuracy") is not None and rnd.get("answer_accuracy") is not None:
        d = rnd["answer_accuracy"] - hn["answer_accuracy"]
        struct_classes = {"wrong_row", "wrong_column", "same_leaf_header", "nearby_cell"}
        struct_hurt = [b["answer_accuracy"] for c, b in hn.get("by_distractor_class", {}).items()
                      if c in struct_classes]
        other_hurt = [b["answer_accuracy"] for c, b in hn.get("by_distractor_class", {}).items()
                     if c not in struct_classes and c != "exact_match"]
        gap2 = (round((sum(other_hurt) / len(other_hurt) if other_hurt else 1)
                     - (sum(struct_hurt) / len(struct_hurt) if struct_hurt else 1), 4))
        if d > 0.05 and struct_hurt:
            h3 = v("Supported" if gap2 > 0.05 else "Partial",
                  f"random distractor 대비 hard-negative 정확도 {round(d,4)} 낮음; "
                  f"구조적 클래스(wrong_row/column 등) 평균 {round(sum(struct_hurt)/len(struct_hurt),4) if struct_hurt else 'n/a'} "
                  f"vs 그 외 {round(sum(other_hurt)/len(other_hurt),4) if other_hurt else 'n/a'}")
        elif d > 0:
            h3 = v("Partial", f"random 대비 {round(d,4)} 낮으나 구조 클래스별 차이는 미약")
        else:
            h3 = v("Rejected", f"random distractor와 hard-negative 정확도 차이 {round(d,4)} — 구조가 특별히 더 방해하지 않음")
    else:
        h3 = v("Undetermined", "gold_1_random / gold_1_hard_negative 레그 데이터 없음")

    return {"H1_predicted_structure_reconstruction_error": h1,
           "H2_embedding_ignores_structure": h2,
           "H3_structural_hard_negative_confuses_reader": h3}


def write_md() -> Path:
    struct = json.loads((OUT_DIR / "structure_metrics.json").read_text()) \
        if (OUT_DIR / "structure_metrics.json").exists() else {}
    decomp = json.loads((OUT_DIR / "structure_decomposition.json").read_text()) \
        if (OUT_DIR / "structure_decomposition.json").exists() else {}
    retr = {r: json.loads((OUT_DIR / f"retrieval_{r}.json").read_text())
           for r in REPRESENTATIONS if (OUT_DIR / f"retrieval_{r}.json").exists()}
    hardneg = json.loads((OUT_DIR / "hardneg_analysis.json").read_text()) \
        if (OUT_DIR / "hardneg_analysis.json").exists() else {}
    vd = verdict(struct, decomp, retr, hardneg) if struct and decomp and retr and hardneg else {}
    unknown = unknown_rank_query_ids()

    lines = ["# 병목 원인 진단 (H1/H2/H3)", "",
             "BOTTLENECK_DIAGNOSIS.md의 후속. ESM 정의는 그대로(단일 gold+고정 k 구조상 k=1에서만 "
             "ESM>0). alpha=0.7 등 기존 하이퍼파라미터는 재튜닝하지 않았고, gold는 채점/oracle "
             "조건 외에는 랭킹에 쓰지 않았다.", "",
             f"**gold_rank 미확인 {len(unknown)}건** (retrieval_accuracy.py 상위-500 스캔 기준, "
             "별도 처리 — 아래 표의 버킷/평균에 섞지 않음): " + ", ".join(unknown), ""]

    lines += ["## H1 — predicted 구조 복원 오류", ""]
    if struct:
        lines += [f"- 재구성 대상 표 {struct['n_tables_total']}개 중 "
                 f"{struct['n_tables_excluded']}개 제외 ({struct['excluded_reasons']})",
                 f"- row path EM {struct['row_path_exact_match']} / col path EM {struct['col_path_exact_match']}",
                 f"- row path-node F1 {struct['row_path_segment_f1_path_node_f1']} / "
                 f"col path-node F1 {struct['col_path_segment_f1_path_node_f1']}",
                 f"- row pair P/R/F1 {struct['row_pair_prf']}", f"- col pair P/R/F1 {struct['col_pair_prf']}", ""]
    else:
        lines += ["skip — predicted structure 계산 불가(필요 필드: raw texts, "
                 "hitab_grid row_map/col_map). 이 저장소에서는 사용 가능해 skip 사유 없음.", ""]
    if decomp:
        lines += ["structure correct/wrong × predicted_path 검색 성능:", "",
                 "| subset | query count | R@1 | R@5 | R@10 | R@20 | MRR | median rank |",
                 "|---|---|---|---|---|---|---|---|"]
        for key, label in (("predicted_path__structure_correct", "structure correct"),
                          ("predicted_path__structure_wrong", "structure wrong"),
                          ("predicted_path__structure_undetermined", "undetermined")):
            b = decomp[key]
            if b.get("n"):
                lines.append(f"| {label} | {b['n']} | {b['recall_at_1']} | {b['recall_at_5']} | "
                            f"{b['recall_at_10']} | {b['recall_at_20']} | {b['mrr']} | {b['median_rank']} |")
        lines += ["", "structure-correct 부분집합에서 gold_path vs predicted_path (재구성이 맞아도 "
                 "남는 차이가 있는지):", "",
                 "| repr | query count | R@1 | R@5 | R@10 | R@20 | MRR | median rank |", "|---|---|---|---|---|---|---|---|"]
        for key, label in (("gold_path__on_structure_correct_subset", "gold_path"),
                          ("predicted_path__on_structure_correct_subset", "predicted_path")):
            b = decomp[key]
            if b.get("n"):
                lines.append(f"| {label} | {b['n']} | {b['recall_at_1']} | {b['recall_at_5']} | "
                            f"{b['recall_at_10']} | {b['recall_at_20']} | {b['mrr']} | {b['median_rank']} |")
        lines.append("")

    lines += ["## H2 — embedding이 구조를 무시하는가", "",
             "| representation | R@1 | R@5 | R@10 | R@20 | MRR | median rank | n_units | cache_hit |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in REPRESENTATIONS:
        d = retr.get(r)
        if d:
            lines.append(f"| {r} | {d['recall_at_1']} | {d['recall_at_5']} | {d['recall_at_10']} | "
                        f"{d['recall_at_20']} | {d['mrr']} | {d['median_rank']} | {d['n_units']} | "
                        f"{d['embed_cache_hit']} |")
    lines.append("")

    lines += ["## H3 — 구조적 hard-negative가 리더를 방해하는가", ""]
    if hardneg:
        lines += [f"gold-only 기준선 답변 정확도: {hardneg['gold_only_baseline_answer_accuracy']} "
                 f"(n_sample={hardneg['n_sample']})", "",
                 "| condition | query count | 답변 정확도 |", "|---|---|---|"]
        for cond, b in hardneg["legs"].items():
            lines.append(f"| {cond} | {b['n']} | {b['answer_accuracy']} |")
        lines.append("")
        for cond in ("gold_1_hard_negative_first", "gold_1_hard_negative_last"):
            b = hardneg["legs"].get(cond)
            if not b:
                continue
            lines += [f"### {cond} — distractor 구조 클래스별", "",
                     "| class | query count | 답변 정확도 |", "|---|---|---|"]
            for c, v in sorted(b["by_distractor_class"].items()):
                lines.append(f"| {c} | {v['n']} | {v['answer_accuracy']} |")
            lines += ["", f"### {cond} — score margin 구간별 (margin은 distractor가 top-1일 때만 계산됨)", "",
                     "| margin bucket | query count | 답변 정확도 |", "|---|---|---|"]
            for m, v in sorted(b["by_margin_bucket"].items()):
                lines.append(f"| {m} | {v['n']} | {v['answer_accuracy']} |")
            lines.append("")

    lines += ["## 판정", ""]
    for h, label in (("H1_predicted_structure_reconstruction_error", "H1 (구조 복원 오류)"),
                    ("H2_embedding_ignores_structure", "H2 (임베딩이 구조 무시)"),
                    ("H3_structural_hard_negative_confuses_reader", "H3 (구조적 hard-negative 방해)")):
        d = vd.get(h)
        if d:
            lines += [f"**{label}: {d['status']}** — {d['why']}", ""]

    lines += ["## 한계", "",
             "- predicted structure는 헤더 블록 크기(nhr/nhc)는 gold를 쓰고 경로 내용만 재구성함 "
             "(tree_reconstruct_hitab_raw.py의 known-boundary 모드와 동일 조건).",
             "- H3 margin은 hard-negative distractor가 해당 질의의 top-1 검색 결과와 정확히 같을 때만 "
             "계산됨 — 그 외는 'unknown' 버킷.",
             "- controlled QA는 query count=150 표본(2026-09-16 seed=42 고정, 재실행 없음)에 대한 재분석이며 "
             "새 리더 호출은 없음.", "",
             "## 산출물", "",
             "- `results/bottleneck_root_cause/structure_metrics.json`",
             "- `results/bottleneck_root_cause/structure_decomposition.json`",
             "- `results/bottleneck_root_cause/retrieval_{value_only,gold_path,predicted_path}.json(l)`",
             "- `results/bottleneck_root_cause/top1_error_detail.csv`",
             "- `results/bottleneck_root_cause/hardneg_analysis.json`", ""]

    out = OUT_DIR / "BOTTLENECK_ROOT_CAUSE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("structure")
    rp = sub.add_parser("retrieval")
    rp.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    sub.add_parser("errorcsv")
    sub.add_parser("hardneg")
    sub.add_parser("verdict")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.cmd == "structure":
        m = structure_metrics(split_corpus_table_ids())
        (OUT_DIR / "structure_metrics.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(m, indent=2, ensure_ascii=False))
    elif a.cmd == "retrieval":
        run_retrieval(a.representation)
        if a.representation == "predicted_path" and (OUT_DIR / "retrieval_gold_path.json").exists():
            d = decompose_by_structure()
            (OUT_DIR / "structure_decomposition.json").write_text(
                json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(d, indent=2, ensure_ascii=False))
    elif a.cmd == "errorcsv":
        p = write_error_csv()
        print(f"wrote -> {p}")
    elif a.cmd == "hardneg":
        d = hardneg_analysis()
        (OUT_DIR / "hardneg_analysis.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(d, indent=2, ensure_ascii=False))
    elif a.cmd == "verdict":
        p = write_md()
        print(f"wrote -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
