#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end header-hierarchy reconstruction WITHOUT gold nhr/nhc
(FULL_STRUCTURE_EVAL.md).

Boundary-guessing code already exists in this repo
(``rag_agent.reconstruct.guess_n_header_rows`` / ``guess_n_header_cols``,
documented and pre-validated in scripts/tree_reconstruct_hitab_raw.py and
header_grid.py's own docstrings) -- this script composes it with the SAME
path-reconstruction algorithm bottleneck_root_cause.py already uses for the
known-boundary condition (``predicted_structure``: gold nhr/nhc, reconstructed
path CONTENT), swapping only the boundary source. Nothing here guesses a new
heuristic; it wires up two already-implemented, already-measured functions
that nothing in the pipeline previously called together.

Boundary bootstrap (no gold nhr/nhc touches this at any point):
  nhc0 = guess_n_header_cols(texts, n_header_rows=1)   # blind default nhr=1
  nhr  = guess_n_header_rows(texts, n_header_cols=nhc0)
  nhc  = guess_n_header_cols(texts, n_header_rows=nhr)  # refine with guessed nhr
Gold nhr/nhc (``min(tab.row_map)``/``min(tab.col_map)``) is read ONLY to
SCORE the guess afterward, never fed into guess_n_header_rows/cols or
reconstruct_{row,col}_paths.

Reused, not reimplemented: hitab_grid.load_table/HitabTable (row_map/col_map
-- which DATA cell a raw line is, from the gold hmt tree, exactly as every
other condition already assumes), reconstruct_row_paths/reconstruct_col_paths,
tree_reconstruct_hitab_raw.{norm,segment_f1,hierarchy_edges},
bottleneck_diagnosis.load_primary_population, bottleneck_root_cause.
{split_corpus_table_ids,encode_corpus}, retrieval_accuracy.cell_unit,
retrieval_improvement.encode_queries_cached, hybrid_index._minmax,
sparse_bm25.SparseBM25, encoders.default_encoder. No file under rag_agent/ or
scripts/{retrieval_accuracy,bottleneck_diagnosis,bottleneck_root_cause}.py is
modified.

  PYTHONPATH=. .venv/bin/python scripts/full_structure_eval.py structure
  PYTHONPATH=. .venv/bin/python scripts/full_structure_eval.py retrieval
  PYTHONPATH=. .venv/bin/python scripts/full_structure_eval.py report
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,  # noqa: E402
                                   reconstruct_col_paths, reconstruct_row_paths)
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import with_page_title           # noqa: E402
from retrieval_accuracy import PAGE_TITLES, cell_unit                  # noqa: E402
from retrieval_improvement import encode_queries_cached                # noqa: E402
from bottleneck_diagnosis import load_primary_population               # noqa: E402
from bottleneck_root_cause import (encode_corpus, split_corpus_table_ids)  # noqa: E402
from tree_reconstruct_hitab_raw import hierarchy_edges, norm, segment_f1  # noqa: E402

DATA_DIR = "data/hitab"
ALPHA = 0.7
OUT_DIR = ROOT / "results/full_structure_eval"
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}

_CACHE: dict = {}


def guess_boundary(texts) -> tuple:
    nhc0 = guess_n_header_cols(texts, n_header_rows=1)
    nhr = guess_n_header_rows(texts, n_header_cols=nhc0)
    nhc = guess_n_header_cols(texts, n_header_rows=nhr)
    return nhr, nhc


def predicted_structure_full(tid: str, data_dir: str = DATA_DIR):
    """``(row_path_fn, col_path_fn, nhr, nhc, meta)``. ``meta`` carries
    ``nhr_gold``/``nhc_gold`` for scoring only -- neither ever reaches
    ``guess_n_header_rows/cols`` or ``reconstruct_*_paths`` above this line."""
    if tid in _CACHE:
        return _CACHE[tid]
    tab = hg.load_table(tid, data_dir)
    if tab is None or not tab.row_map or not tab.col_map:
        out = (None, None, None, None, "no_raw_or_unmapped_axis")
        _CACHE[tid] = out
        return out
    texts = tab.raw.get("texts") or []
    nhr_gold, nhc_gold = min(tab.row_map), min(tab.col_map)
    if nhr_gold <= 0 or nhc_gold <= 0 or not texts:
        out = (None, None, None, None, "degenerate_header_block")
        _CACHE[tid] = out
        return out
    nhr, nhc = guess_boundary(texts)
    if nhr <= 0 or nhc <= 0:
        out = (None, None, nhr, nhc, {"nhr_gold": nhr_gold, "nhc_gold": nhc_gold,
                                      "reason": "degenerate_guessed_boundary"})
        _CACHE[tid] = out
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

    out = (row_path, col_path, nhr, nhc, {"nhr_gold": nhr_gold, "nhc_gold": nhc_gold})
    _CACHE[tid] = out
    return out


def cell_structure_correct_full(tid, i, j, data_dir=DATA_DIR):
    row_fn, col_fn, _nhr, _nhc, meta = predicted_structure_full(tid, data_dir)
    if row_fn is None or isinstance(meta, str) or "reason" in (meta or {}):
        return None
    t = hg.load_table(tid, data_dir).table
    rp, cp = row_fn(i), col_fn(j)
    if rp is None or cp is None:
        return None
    return norm(rp) == norm(t.row_path(i)) and norm(cp) == norm(t.col_path(j))


# ---------------------------------------------------------------------------
# structure metrics (boundary accuracy, path EM/F1, pair F1, joint EM)
# ---------------------------------------------------------------------------

def run_structure() -> dict:
    tids = split_corpus_table_ids()
    boundary_rows = []
    col_hit = col_tot = row_hit = row_tot = 0
    col_f1, row_f1 = [], []
    pairs = {"col": [0, 0, 0], "row": [0, 0, 0]}
    excluded = Counter()

    for tid in tids:
        row_fn, col_fn, nhr, nhc, meta = predicted_structure_full(tid, DATA_DIR)
        if row_fn is None:
            excluded[meta if isinstance(meta, str) else meta.get("reason", "excluded")] += 1
            continue
        boundary_rows.append({"table_id": tid, "nhr_guess": nhr, "nhc_guess": nhc,
                              "nhr_gold": meta["nhr_gold"], "nhc_gold": meta["nhc_gold"]})
        t = hg.load_table(tid, DATA_DIR).table
        gold_rows, rec_rows_ = [], []
        for i in range(t.n_rows):
            rp = row_fn(i)
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
            cp = col_fn(j)
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

    def prf(hit, n_rec, n_gold):
        p = hit / n_rec if n_rec else 0.0
        r = hit / n_gold if n_gold else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4),
               "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
               "n_gold_edges": n_gold, "n_rec_edges": n_rec}

    def m(xs):
        return round(mean(xs), 4) if xs else None

    n_nhr_exact = sum(1 for b in boundary_rows if b["nhr_guess"] == b["nhr_gold"])
    n_nhc_exact = sum(1 for b in boundary_rows if b["nhc_guess"] == b["nhc_gold"])
    nhr_mae = m([abs(b["nhr_guess"] - b["nhr_gold"]) for b in boundary_rows])
    nhc_mae = m([abs(b["nhc_guess"] - b["nhc_gold"]) for b in boundary_rows])

    pop = load_primary_population()
    joint = {qid: cell_structure_correct_full(*sorted(r["gold_cells"])[0], DATA_DIR)
            for qid, r in pop.items()}
    n_joint_correct = sum(1 for v in joint.values() if v is True)
    n_joint_wrong = sum(1 for v in joint.values() if v is False)
    n_joint_undetermined = sum(1 for v in joint.values() if v is None)

    out = {
        "n_tables_total": len(tids), "n_tables_excluded": sum(excluded.values()),
        "excluded_reasons": dict(excluded),
        "boundary": {"n_tables_scored": len(boundary_rows),
                    "nhr_exact_accuracy": round(n_nhr_exact / len(boundary_rows), 4),
                    "nhc_exact_accuracy": round(n_nhc_exact / len(boundary_rows), 4),
                    "nhr_mae": nhr_mae, "nhc_mae": nhc_mae,
                    "bootstrap": "nhc0=guess_cols(nhr=1); nhr=guess_rows(nhc=nhc0); "
                                "nhc=guess_cols(nhr=nhr); gold never used as input"},
        "row_path_exact_match": round(row_hit / row_tot, 4) if row_tot else None,
        "col_path_exact_match": round(col_hit / col_tot, 4) if col_tot else None,
        "row_path_segment_f1_path_node_f1": m(row_f1),
        "col_path_segment_f1_path_node_f1": m(col_f1),
        "row_pair_prf": prf(*pairs["row"]), "col_pair_prf": prf(*pairs["col"]),
        "row_paths_scored": row_tot, "col_paths_scored": col_tot,
        "joint_path_em_primary_population": {
            "n_population": len(pop), "n_correct": n_joint_correct,
            "n_wrong": n_joint_wrong, "n_undetermined": n_joint_undetermined,
            "rate": round(n_joint_correct / len(pop), 4)},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "boundary_detail.json").write_text(
        json.dumps(boundary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "structure_metrics_full.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "joint_full_by_query.json").write_text(
        json.dumps(joint, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# retrieval (R@1/5/20, MRR, ESM) using the fully-predicted-path corpus
# ---------------------------------------------------------------------------

def build_fully_predicted_corpus(data_dir: str = DATA_DIR):
    tids = split_corpus_table_ids()
    texts, coords, fallback = [], [], 0
    for tid in tids:
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
        row_fn, col_fn, _nhr, _nhc, meta = predicted_structure_full(tid, data_dir)
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = t.data[i][j]
                if not str(v).strip():
                    continue
                rp = row_fn(i) if row_fn else None
                cp = col_fn(j) if col_fn else None
                if rp is None or cp is None:
                    txt = str(v)
                    fallback += 1
                else:
                    txt = cell_unit(title, rp, cp, v, "s3c")
                texts.append(txt)
                coords.append((tid, i, j))
    return texts, coords, fallback


def run_retrieval() -> dict:
    pop = load_primary_population()
    texts, coords, fallback = build_fully_predicted_corpus()
    coord_index = {c: idx for idx, c in enumerate(coords)}
    if len(coord_index) != len(coords):
        raise ValueError("cell-unit corpus has a duplicate coordinate")

    enc = default_encoder()
    emb, cache_hit = encode_corpus(texts, enc)
    bm = SparseBM25(_tokenize(t) for t in texts)

    qids = sorted(pop)
    qtexts = [pop[q]["question"] for q in qids]
    qemb = encode_queries_cached(qtexts, enc)

    rows = []
    for k, qid in enumerate(qids):
        r = pop[qid]
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            rows.append({"query_id": qid, "excluded": "gold_cell_not_in_corpus"})
            continue
        dense = emb @ qemb[k].astype(np.float32)
        sparse = bm.get_scores(_tokenize(r["question"]))
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
        order = np.argsort(-hybrid, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        gold_rank = int(ranks[gidx])
        top1_coord = coords[int(order[0])]
        rows.append({"query_id": qid, "gold_cell": list(gold), "top1_cell": list(top1_coord),
                    "gold_rank": gold_rank, "esm": int(gold_rank == 1 and top1_coord == gold)})

    scored = [r for r in rows if "excluded" not in r]
    ranks_arr = [r["gold_rank"] for r in scored]

    def recall_at(k):
        return round(sum(x <= k for x in ranks_arr) / len(ranks_arr), 4) if ranks_arr else None

    summary = {"representation": "fully_predicted_path", "alpha": ALPHA, "encoder": enc.name,
              "embed_cache_hit": cache_hit, "n_units": len(texts),
              "n_fallback_to_value_only": fallback,
              "n_population": len(pop), "n_scored": len(scored), "n_excluded": len(rows) - len(scored),
              "recall_at_1": recall_at(1), "recall_at_5": recall_at(5),
              "recall_at_10": recall_at(10), "recall_at_20": recall_at(20),
              "esm_at_1": recall_at(1),
              "mrr": round(sum(1 / x for x in ranks_arr) / len(ranks_arr), 4) if ranks_arr else None}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_fully_predicted_path.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT_DIR / "retrieval_fully_predicted_path.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def write_report() -> Path:
    sm_gold_path = "trivial: paths ARE gold by construction"
    known = json.loads((ROOT / "results/bottleneck_root_cause/structure_metrics.json").read_text())
    known_ret = json.loads((ROOT / "results/bottleneck_root_cause/retrieval_predicted_path.json").read_text())
    known_decomp = json.loads((ROOT / "results/bottleneck_root_cause/structure_decomposition.json").read_text())
    gold_ret = json.loads((ROOT / "results/bottleneck_root_cause/retrieval_gold_path.json").read_text())
    full_sm = json.loads((OUT_DIR / "structure_metrics_full.json").read_text())
    full_ret = json.loads((OUT_DIR / "retrieval_fully_predicted_path.json").read_text())

    known_joint = known_decomp["n_structure_correct"]
    known_joint_n = known_decomp["n_structure_correct"] + known_decomp["n_structure_wrong"]

    lines = ["# 완전 계층구조 복원 평가 (FULL_STRUCTURE_EVAL)", "",
             "gold nhr/nhc는 채점에만 사용, 복원 입력(경계 추정/경로 재구성)에는 전혀 사용하지 않음. "
             "경계 추정은 이미 구현·검증된 `rag_agent.reconstruct.guess_n_header_rows/guess_n_header_cols`를 "
             "재사용(새 휴리스틱 없음). 원본 파일(rag_agent/, scripts/{retrieval_accuracy,bottleneck_diagnosis,"
             "bottleneck_root_cause}.py) 미수정 — 신규 스크립트 `scripts/full_structure_eval.py`만 추가.", "",
             "## 경계 부트스트랩 절차 (gold 미사용)", "",
             "```", full_sm["boundary"]["bootstrap"], "```", "",
             "## 1) header boundary (nhr/nhc), fully-predicted-path만 해당", "",
             f"- 대상 table 수: {full_sm['boundary']['n_tables_scored']} / {full_sm['n_tables_total']} "
             f"(제외 {full_sm['n_tables_excluded']}: {full_sm['excluded_reasons']})",
             f"- nhr exact accuracy: {full_sm['boundary']['nhr_exact_accuracy']}, MAE={full_sm['boundary']['nhr_mae']}",
             f"- nhc exact accuracy: {full_sm['boundary']['nhc_exact_accuracy']}, MAE={full_sm['boundary']['nhc_mae']}",
             "- gold-path/known-boundary 조건은 nhr/nhc가 gold이므로 이 항목이 정의상 100%/0 — "
             "비교 대상 아님(fully-predicted-path만 해당하는 항목)", "",
             "## 2) 비교표 — path EM / F1 / pair F1 / joint EM", "",
             "| condition | row EM | col EM | row F1 | col F1 | row pair F1 | col pair F1 | joint EM(gold cell, query count=991) |",
             "|---|---|---|---|---|---|---|---|",
             f"| gold-path | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 (991/991, {sm_gold_path}) |",
             f"| known-boundary predicted-path | {known['row_path_exact_match']} | {known['col_path_exact_match']} | "
             f"{known['row_path_segment_f1_path_node_f1']} | {known['col_path_segment_f1_path_node_f1']} | "
             f"{known['row_pair_prf']['f1']} | {known['col_pair_prf']['f1']} | "
             f"{round(known_joint / known_joint_n, 4)} ({known_joint}/{known_joint_n}) |",
             f"| fully-predicted-path | {full_sm['row_path_exact_match']} | {full_sm['col_path_exact_match']} | "
             f"{full_sm['row_path_segment_f1_path_node_f1']} | {full_sm['col_path_segment_f1_path_node_f1']} | "
             f"{full_sm['row_pair_prf']['f1']} | {full_sm['col_pair_prf']['f1']} | "
             f"{full_sm['joint_path_em_primary_population']['rate']} "
             f"({full_sm['joint_path_em_primary_population']['n_correct']}/"
             f"{full_sm['joint_path_em_primary_population']['n_population']}) |",
             "", "## 3) Hybrid retrieval (R@1/5/20, MRR, ESM), test query count=991", "",
             "| condition | R@1 | R@5 | R@20 | MRR | ESM |", "|---|---|---|---|---|---|",
             f"| gold-path | {gold_ret['recall_at_1']} | {gold_ret['recall_at_5']} | {gold_ret['recall_at_20']} | "
             f"{gold_ret['mrr']} | {gold_ret['recall_at_1']} |",
             f"| known-boundary predicted-path | {known_ret['recall_at_1']} | {known_ret['recall_at_5']} | "
             f"{known_ret['recall_at_20']} | {known_ret['mrr']} | {known_ret['recall_at_1']} |",
             f"| fully-predicted-path | {full_ret['recall_at_1']} | {full_ret['recall_at_5']} | "
             f"{full_ret['recall_at_20']} | {full_ret['mrr']} | {full_ret['esm_at_1']} |",
             "", f"fully-predicted-path fallback-to-value-only 셀 수: {full_ret['n_fallback_to_value_only']} "
             f"/ {full_ret['n_units']}", "",
             "## 한계", "",
             "- 경계 부트스트랩은 1라운드 상호 보정만 수행(2회 이상 반복 시 소폭 달라질 수 있음, 미검증).",
             "- `guess_n_header_cols`는 header_grid.py 자체 문서 기준 test 91.8% exact(다른 실험/데이터 경로, "
             "2026-08-26 측정) — 이 표의 nhc 정확도는 본 실험(split corpus 538 tables, retrieval용 정의)에서 "
             "다시 측정한 것으로 그 수치와 다를 수 있음.",
             "- tree_reconstruct_hitab_raw.py 모듈 docstring은 \"n_header_cols용 guesser 없음\"이라고 적혀 있으나 "
             "실제로는 guess_n_header_cols가 이미 구현되어 있음(문서가 stale) — 원본 미수정 원칙에 따라 "
             "docstring은 고치지 않음, 여기 기록만 남김.",
             "- row_map/col_map(어느 raw 줄이 데이터 셀인지)은 gold hmt 트리에서 유도된 매핑을 그대로 사용 "
             "(gold-path/known-boundary 조건과 동일 전제) — 이는 세 조건 모두가 이미 공유하는 전제이며, "
             "nhr/nhc 자체를 gold에서 가져오는 것과는 다르다.", ""]

    out = ROOT / "results/full_structure_eval/FULL_STRUCTURE_EVAL.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("structure")
    sub.add_parser("retrieval")
    sub.add_parser("report")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "structure":
        run_structure()
    elif a.cmd == "retrieval":
        run_retrieval()
    elif a.cmd == "report":
        p = write_report()
        print(f"wrote -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
