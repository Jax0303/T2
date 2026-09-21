#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retrieval-level test of rag_agent.serialization.templates.STRUCTURAL_LEAF
(row/col leaf tokens repeated as a sentence prefix) against the deployed s3c
(STRUCTURAL_COMPACT) baseline, on the SAME primary population, encoder and
hybrid formula (alpha=0.7) as results/bottleneck_root_cause/. Reuses
scripts/bottleneck_root_cause.py's corpus/retrieval loop verbatim except for
the per-cell template, and scripts/bottleneck_diagnosis.py's population loader
and top-1 error classifier. No training, no encoder change.

  PYTHONPATH=. .venv/bin/python scripts/sentence_disambiguation_eval.py retrieval
  PYTHONPATH=. .venv/bin/python scripts/sentence_disambiguation_eval.py errorcsv
  PYTHONPATH=. .venv/bin/python scripts/sentence_disambiguation_eval.py report
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.artifacts import digest, file_digest              # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize        # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import caption_sentence, with_page_title  # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_LEAF, STRUCTURAL_LEAF_X2  # noqa: E402
from scripts.retrieval_accuracy import PAGE_TITLES                    # noqa: E402
from bottleneck_diagnosis import classify_top1_error, load_primary_population, split_corpus_table_ids  # noqa: E402

OUT_DIR = ROOT / "results/sentence_disambiguation_20260916"
BASELINE_DIR = ROOT / "results/bottleneck_root_cause"
EMBED_CACHE = ROOT / ".cache/retrieval_accuracy"
ALPHA = 0.7
TEMPLATE_BY_NAME = {"structural_leaf": STRUCTURAL_LEAF, "structural_leaf_x2": STRUCTURAL_LEAF_X2}
_PAGE_TITLES = json.loads(PAGE_TITLES.read_text()) if PAGE_TITLES.exists() else {}


def _retrieval_path(ext: str, template_name: str) -> Path:
    return OUT_DIR / f"retrieval_{template_name}.{ext}"


def _out(name_root: str, ext: str, template_name: str) -> Path:
    """structural_leaf keeps the original unsuffixed filename (backward
    compatible with results/sentence_disambiguation_20260916/ already cited
    elsewhere); any other template gets its name appended so runs don't clobber
    each other."""
    if template_name == "structural_leaf":
        return OUT_DIR / f"{name_root}.{ext}"
    return OUT_DIR / f"{name_root}_{template_name}.{ext}"


def build_leaf_corpus(data_dir: str = "data/hitab", template_name: str = "structural_leaf"):
    """Same units/order as bottleneck_root_cause.build_repr_corpus(..., 'gold_path'),
    the given template instead of s3c — the only variable changed."""
    template = TEMPLATE_BY_NAME[template_name]
    texts, coords = [], []
    for tid in split_corpus_table_ids():
        tab = hg.load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = t.data[i][j]
                if not str(v).strip():
                    continue
                texts.append(caption_sentence(title, t.row_path(i), t.col_path(j),
                                              value=v, template=template))
                coords.append((tid, i, j))
    return texts, coords


def encode_corpus(texts, encoder):
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    key = digest({"texts": texts, "encoder": encoder.metadata(), "overflow": "error"})[:24]
    f = EMBED_CACHE / f"{encoder.name.replace('/', '_')}_{len(texts)}_{key}.npy"
    if f.exists():
        return np.load(f), True
    emb = encoder.encode(texts)
    if emb.ndim != 2 or len(emb) != len(texts) or not np.isfinite(emb).all():
        raise ValueError("encoder produced invalid document vectors")
    np.save(f, emb)
    return emb, False


def run_retrieval(data_dir: str = "data/hitab", template_name: str = "structural_leaf") -> dict:
    pop = load_primary_population()
    texts, coords = build_leaf_corpus(data_dir, template_name)
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
            "hybrid_score_gold": float(hybrid[gidx]), "hybrid_score_top1": float(hybrid[top1]),
        })

    scored = [r for r in rows if "excluded" not in r]
    ranks_arr = [r["gold_rank"] for r in scored]

    def recall_at(k):
        return round(sum(x <= k for x in ranks_arr) / len(ranks_arr), 4) if ranks_arr else None

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else None

    summary = {
        "template": template_name, "alpha": ALPHA, "encoder": enc.name,
        "embed_cache_hit": cache_hit, "n_units": len(texts),
        "n_population": len(pop), "n_scored": len(scored), "n_excluded": len(rows) - len(scored),
        "recall_at_1": recall_at(1), "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10), "recall_at_20": recall_at(20),
        "mrr": round(sum(1.0 / x for x in ranks_arr) / len(ranks_arr), 4) if ranks_arr else None,
        "median_rank": median(ranks_arr),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _retrieval_path("jsonl", template_name).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    _retrieval_path("json", template_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def write_error_csv(data_dir: str = "data/hitab", template_name: str = "structural_leaf") -> Path:
    pop = load_primary_population()
    rows_by_qid = {}
    with _retrieval_path("jsonl", template_name).open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            rows_by_qid[r["query_id"]] = r
    tabs: dict = {}
    out_path = _out("top1_error_detail", "csv", template_name)
    counts = Counter()
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "gold_rank", "esm", "gold_table", "gold_row", "gold_col",
                   "top1_table", "top1_row", "top1_col", "error_class"])
        for qid in sorted(pop):
            r = rows_by_qid.get(qid)
            if r is None or "excluded" in r:
                continue
            gtid, gi, gj = r["gold_cell"]
            ttid, ti, tj = r["top1_cell"]
            cls = ("exact_match" if (gtid, gi, gj) == (ttid, ti, tj)
                  else classify_top1_error((ttid, ti, tj), (gtid, gi, gj), tabs, data_dir))
            counts[cls] += 1
            w.writerow([qid, r["gold_rank"], r["esm"], gtid, gi, gj, ttid, ti, tj, cls])
    _out("error_class_counts", "json", template_name).write_text(
        json.dumps(dict(counts), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(counts), indent=2, ensure_ascii=False))
    return out_path


def build_answer_records(data_dir: str = "data/hitab", template_name: str = "structural_leaf") -> Path:
    """A *_records.jsonl compatible with scripts/answer_accuracy.py --condition
    retrieved --k 1: same 991-query primary population, k=1 context replaced
    with the given template's top-1 cell's own rendered sentence. Everything
    query-level (question/gold_cells/mode/m/table_id/answer) is copied
    unchanged from the deployed s3c_v2_records.jsonl -- only the
    retrieval-dependent fields are recomputed."""
    from rag_agent.eval.artifacts import read_records, validate_retrieval

    template = TEMPLATE_BY_NAME[template_name]
    orig = read_records(ROOT / "results/evaluation_v2/s3c_v2_records.jsonl")
    leaf_rows = {}
    with _retrieval_path("jsonl", template_name).open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            leaf_rows[r["query_id"]] = r

    pop = load_primary_population()
    tabs: dict = {}
    out_path = _out("structural_leaf_k1_records", "jsonl", template_name)
    with out_path.open("w", encoding="utf-8") as fh:
        for qid in sorted(pop):
            base = orig[qid]
            lr = leaf_rows[qid]
            tid, i, j = lr["top1_cell"]
            tab = tabs.get(tid) or hg.load_table(tid, data_dir)
            tabs[tid] = tab
            title = with_page_title(tab.title, _PAGE_TITLES.get(tid))
            text = caption_sentence(title, tab.table.row_path(i), tab.table.col_path(j),
                                    value=tab.table.data[i][j], template=template)
            cell = [tid, i, j]
            ctx = [text]
            units = [{"index_unit": 0, "text": text, "cells": [cell]}]
            gold = set(map(tuple, base["gold_cells"]))
            cells = {tuple(cell)}
            hit = gold <= cells if base["mode"] == "all" else bool(gold & cells)
            row = dict(base)
            row.update({
                "context": ctx, "context_units": units, "context_cells": [cell],
                "context_sha256": digest(ctx), "evidence_sha256": digest(units),
                "cells_in_context": 1, "gold_rank": lr["gold_rank"],
                "gold_table_in_context": int(tid == base["table_id"]),
                "correct": int(hit),
            })
            validate_retrieval(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path = out_path.with_name(out_path.stem.removesuffix("_records") + ".json")
    meta_path.write_text(json.dumps({
        "records_sha256": file_digest(out_path), "context_version": 2, "unit": "cell",
        "note": f"{template_name} template, k=1, derived from s3c_v2_records.jsonl "
                "for scripts/answer_accuracy.py --condition retrieved --k 1",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_path, "and", meta_path)
    return out_path


def _load_counts(csv_path: Path) -> Counter:
    counts = Counter()
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts[row["error_class"]] += 1
    return counts


def write_report(template_name: str = "structural_leaf") -> Path:
    base_gold = json.loads((BASELINE_DIR / "retrieval_gold_path.json").read_text())
    base_pred = json.loads((BASELINE_DIR / "retrieval_predicted_path.json").read_text())
    base_counts = _load_counts(BASELINE_DIR / "top1_error_detail.csv")

    # (label, R@k summary dict, error-class Counter) rows, baseline first, current template last.
    ladder = [("gold_path (s3c, baseline)", base_gold, base_counts),
             ("predicted_path (s3c, baseline)", base_pred, None)]
    leaf_json = _retrieval_path("json", "structural_leaf")
    if template_name != "structural_leaf" and leaf_json.exists():
        ladder.append(("structural_leaf (one leaf-prefix repeat)",
                       json.loads(leaf_json.read_text()),
                       _load_counts(_out("top1_error_detail", "csv", "structural_leaf"))))
    ladder.append((f"{template_name} (new)",
                  json.loads(_retrieval_path("json", template_name).read_text()),
                  _load_counts(_out("top1_error_detail", "csv", template_name))))

    lines = [f"# 문장 축-구분 개입 ({template_name}) — 2026-09-17", "",
            "모집단: hitab test primary (mode=all, m=1, aggregation=none), query count=991. "
            "encoder/hybrid(alpha=0.7)/corpus(split)는 baseline과 동일 — 바뀐 변수는 "
            "cell 문장 템플릿 하나뿐.", "",
            f"## R@1/5/10/20/MRR — baseline .. {template_name}", "",
            "| repr | R@1 | R@5 | R@10 | R@20 | MRR | median rank |",
            "|---|---|---|---|---|---|---|"]
    for label, r, _ in ladder:
        lines.append(f"| {label} | {r['recall_at_1']} | {r['recall_at_5']} | "
                     f"{r['recall_at_10']} | {r['recall_at_20']} | {r['mrr']} | {r['median_rank']} |")

    err_rows = [(label, c) for label, _, c in ladder if c is not None]
    lines += ["", f"## top-1 오류 클래스 — baseline(gold_path, s3c) .. {template_name}", "",
             "| class | " + " | ".join(f"{label} n | {label} %" for label, _ in err_rows) + " |",
             "|---|" + "---|" * (2 * len(err_rows))]
    totals = [sum(c.values()) - c.get("exact_match", 0) for _, c in err_rows]
    for cls in ("same_value", "wrong_row", "wrong_column", "same_leaf_header",
               "nearby_cell", "wrong_table", "other"):
        cells = []
        for (_, c), total in zip(err_rows, totals):
            n = c.get(cls, 0)
            pct = round(n / total, 4) if total else None
            cells.append(f"{n} | {pct}")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    lines.append("| **total errors (non-exact-match)** | " +
                 " | ".join(f"{t} | 1.0" for t in totals) + " |")
    out = _out("SENTENCE_DISAMBIGUATION", "md", template_name)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("retrieval", "errorcsv", "answerrecords", "report"))
    ap.add_argument("--template", choices=tuple(TEMPLATE_BY_NAME), default="structural_leaf")
    args = ap.parse_args()
    if args.cmd == "retrieval":
        run_retrieval(template_name=args.template)
    elif args.cmd == "errorcsv":
        write_error_csv(template_name=args.template)
    elif args.cmd == "answerrecords":
        build_answer_records(template_name=args.template)
    else:
        write_report(template_name=args.template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
