#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Header-path coverage (HPC): does the chunk that HOLDS a gold cell also hold
that cell's header ancestors, as literal text?

Descriptive only. One row per (query, gold cell, chunk-that-contains-it):

    full coverage    every element of header_path(c) occurs in the chunk text
    partial ratio    |elements present| / |header_path(c)|

header_path(c) = row-header ancestors ++ column-header ancestors, in that
order, taken from the SAME paths the retrieval corpus was built from
(``Corpus.cell_paths`` in scripts/corpus_dump_vs_cell.py), so a coverage number
here refers to the paths the index actually carries, not a second parse.

Matching is substring, after whitespace collapse + lowercasing, and nothing
else. No stemming, no synonyms.

Chunking policies (``--policy``):

  P1_fixed_256/512/1024  the table's markdown cut into fixed token windows
  P2_row                 one chunk per data row: row path + TOP-LEVEL column
                         header per cell (NOT the repo's existing row arm,
                         which uses the LEAF column header -- see RUN_LOG)
  P3_whole_table         the whole table markdown as one chunk
  P4_path_cell           one chunk per cell, the deployed sentence (--cell-scheme)

Run:
  PYTHONPATH=. python3 analysis/header_path_coverage.py --dataset hitab \
      --policy P4_path_cell --max-queries 50
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import corpus_dump_vs_cell as cdv                                    # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,   # noqa: E402
                                               STRUCTURAL_COMPACT)

POLICIES = ("P1_fixed_256", "P1_fixed_512", "P1_fixed_1024",
            "P2_row", "P3_whole_table", "P4_path_cell")
TEMPLATE = {"S3": STRUCTURAL, "S3c": STRUCTURAL_COMPACT, "mt2net": MT2NET}


def norm(s) -> str:
    """Whitespace collapse + lowercase. The ONLY normalization allowed here."""
    return " ".join(str(s).split()).lower()


def header_path(rp, cp) -> list:
    """row-header ancestors ++ column-header ancestors, empties dropped."""
    return [str(x).strip() for x in list(rp) + list(cp) if str(x).strip()]


# --------------------------------------------------------------------------
# where each cell's value sits inside the table's markdown, in characters
# --------------------------------------------------------------------------
def cell_spans(C, tid: str, have: dict) -> dict:
    """(i, j) -> (start, end) char offsets of that cell's field in the markdown.

    Every corpus builder in corpus_dump_vs_cell emits md_lines as
    ``header lines + separator + one line per DATA row``, so data row i is the
    i-th line from the end minus n_r. A line may carry leading header columns
    (RealHiTBench/MultiHiertt keep them); the count is recovered as
    ``fields - n_c`` rather than per-dataset.
    """
    lines, (n_r, n_c) = C.md_lines[tid], C.shape[tid]
    if len(lines) < n_r or n_r == 0:
        return {}
    line_off, base = [], 0
    for ln in lines:
        line_off.append(base)
        base += len(ln) + 1                      # "\n".join
    spans, first = {}, len(lines) - n_r
    for i in range(n_r):
        parts = lines[first + i].split("|")
        if len(parts) < 3:
            continue
        off = (len(parts) - 2) - n_c             # leading header columns
        if off < 0:
            continue
        starts, pos = [], 0
        for p in parts:
            starts.append(pos)
            pos += len(p) + 1
        for j in range(n_c):
            if (i, j) not in have:
                continue
            f = off + j + 1                      # parts[0] precedes the first '|'
            spans[(i, j)] = (line_off[first + i] + starts[f],
                             line_off[first + i] + starts[f] + len(parts[f]))
    return spans


# --------------------------------------------------------------------------
# policies: a chunk is {chunk_id, text, cells}
# --------------------------------------------------------------------------
def chunks_for(C, tid, policy, bud, have, scheme):
    if policy == "P3_whole_table":
        return [{"chunk_id": f"{tid}::whole", "text": "\n".join(C.md_lines[tid]),
                 "cells": set(have)}]

    if policy == "P4_path_cell":
        out = []
        for (i, j), n in have.items():
            rp, cp, v = C.cell_paths[n]
            txt = (cdv.cell_text(rp, cp, v, "S2") if scheme == "S2" else
                   caption_sentence(C.title.get(tid, ""), rp, cp, v,
                                    template=TEMPLATE[scheme]))
            out.append({"chunk_id": f"{tid}::cell::{i}:{j}", "text": txt,
                        "cells": {(i, j)}})
        return out

    if policy == "P2_row":
        rows = defaultdict(list)
        for (i, j) in have:
            rows[i].append(j)
        out = []
        for i in sorted(rows):
            rp = C.cell_paths[have[(i, min(rows[i]))]][0]
            cells = []
            for j in sorted(rows[i]):
                _rp, cp, v = C.cell_paths[have[(i, j)]]
                cells.append(f"{cp[0] if cp else ''}: {v}")     # TOP-LEVEL header
            joined = " | ".join(cells)
            out.append({"chunk_id": f"{tid}::row::{i}",
                        "text": (f"{' > '.join(rp)} | {joined}" if rp else joined),
                        "cells": {(i, j) for j in rows[i]}})
        return out

    n_tok = int(policy.rsplit("_", 1)[1])
    text = "\n".join(C.md_lines[tid])
    offs = bud.tok(text, add_special_tokens=False,
                   return_offsets_mapping=True)["offset_mapping"]
    spans, out = cell_spans(C, tid, have), []
    for k in range(0, len(offs), n_tok):
        win = offs[k:k + n_tok]
        a, b = win[0][0], win[-1][1]
        out.append({"chunk_id": f"{tid}::fixed{n_tok}::{k // n_tok}",
                    "text": text[a:b],
                    "cells": {ij for ij, (s, e) in spans.items() if a <= s and e <= b}})
    return out


def load_corpus(a):
    if a.dataset == "hitab":
        return cdv.hitab_corpus(a.data_dir, a.split, a.population, seed=a.seed)
    if a.dataset == "aitqa":
        return cdv.aitqa_corpus()
    if a.dataset == "realhitbench":
        return cdv.realhitbench_corpus(
            question_types=tuple(a.rhb_question_types),
            require_gold_cells=not a.rhb_em_only,
            population=(a.population if a.population.startswith("rhb") else ""))
    return cdv.multihiertt_corpus(a.mh_queries, a.seed)


def run(C, pop, dataset, policy, bud, scheme, out_dir, verify, seed):
    by_table = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table[t][(i, j)] = n

    cache, rows = {}, []
    n_empty_path = n_not_in_corpus = n_missing = 0
    for q in pop:
        for (tid, i, j) in sorted(q["gold_cells"]):
            have = by_table.get(tid)
            if have is None or (i, j) not in have:
                n_not_in_corpus += 1        # gold cell outside the indexed set
                continue
            rp, cp, v = C.cell_paths[have[(i, j)]]
            path = header_path(rp, cp)
            if not path:
                n_empty_path += 1           # the cell IS a header; excluded
                continue
            if tid not in cache:
                cache[tid] = chunks_for(C, tid, policy, bud, have, scheme)
            hits = [ch for ch in cache[tid] if (i, j) in ch["cells"]]
            if not hits:
                n_missing += 1
                rows.append([dataset, q["query_id"], tid, i, j, len(path),
                             "NOT_IN_ANY_CHUNK", 0, 0.0, False, 0])
                continue
            for ch in hits:
                nt = norm(ch["text"])
                present = sum(1 for e in path if norm(e) in nt)
                rows.append([dataset, q["query_id"], tid, i, j, len(path),
                             ch["chunk_id"], present, present / len(path),
                             present == len(path), bud.count(ch["text"])])

    out = Path(out_dir) / f"{dataset}_{policy}_records.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "query_id", "table_id", "gold_row", "gold_col",
                    "header_path_len", "chunk_id", "path_elems_present",
                    "partial_ratio", "full_coverage", "chunk_token_len"])
        w.writerows(rows)

    scored = [r for r in rows if r[6] != "NOT_IN_ANY_CHUNK"]
    summ = {"dataset": dataset, "policy": policy, "cell_scheme": scheme,
            "n_queries": len(pop), "n_rows": len(rows),
            "n_gold_cells_scored": len({(r[1], r[2], r[3], r[4]) for r in scored}),
            "excluded_empty_header_path": n_empty_path,
            "gold_cell_not_in_corpus_cellset": n_not_in_corpus,
            "not_in_any_chunk": n_missing,
            "HPC_full": (sum(r[9] for r in scored) / len(scored)) if scored else None,
            "HPC_partial": (sum(r[8] for r in scored) / len(scored)) if scored else None,
            "csv": str(out)}
    (out.with_name(f"{dataset}_{policy}_summary.json")).write_text(
        json.dumps(summ, indent=2))

    if verify:
        rng = random.Random(seed)
        for r in rng.sample(scored, min(verify, len(scored))):
            tid, i, j = r[2], r[3], r[4]
            rp, cp, v = C.cell_paths[by_table[tid][(i, j)]]
            ch = next(c for c in cache[tid] if c["chunk_id"] == r[6])
            print("\n" + "=" * 78)
            print(f"[{policy}] query={r[1]} table={tid} cell=({i},{j})")
            print(f"  gold cell text : {v!r}")
            print(f"  header_path    : {header_path(rp, cp)}")
            print(f"  present/total  : {r[7]}/{r[5]}  full={r[9]}")
            print(f"  chunk_id       : {r[6]}  ({r[10]} tok)")
            print("  --- chunk text (verbatim) ---")
            print(ch["text"])
            print("  --- end chunk ---")
    return summ, rows


def selftest():
    """Span math + coverage on a hand-built two-column table."""
    class C:                                    # minimal stand-in for Corpus
        md_lines = {"t": ["| h | Y2020 | Y2021 |", "|---|---|---|",
                          "| Revenue | 10 | 20 |"]}
        shape = {"t": (1, 2)}
        cell_paths = [(["Revenue"], ["Y2020"], "10"), (["Revenue"], ["Y2021"], "20")]
        cell_owner = [("t", 0, 0), ("t", 0, 1)]
        title = {"t": ""}
    have = {(0, 0): 0, (0, 1): 1}
    sp = cell_spans(C, "t", have)
    text = "\n".join(C.md_lines["t"])
    assert text[slice(*sp[(0, 0)])].strip() == "10", text[slice(*sp[(0, 0)])]
    assert text[slice(*sp[(0, 1)])].strip() == "20"
    whole = chunks_for(C, "t", "P3_whole_table", None, have, "S3c")[0]
    p = header_path(*C.cell_paths[0][:2])
    assert sum(norm(e) in norm(whole["text"]) for e in p) == 2   # both headers there
    row = chunks_for(C, "t", "P2_row", None, have, "S3c")[0]
    assert row["text"] == "Revenue | Y2020: 10 | Y2021: 20", row["text"]
    cell = chunks_for(C, "t", "P4_path_cell", None, have, "S2")[0]
    assert cell["text"] == "Revenue > Y2020: 10", cell["text"]
    print("selftest ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "multihiertt", "aitqa", "realhitbench"])
    ap.add_argument("--policy", default="P4_path_cell",
                    help=f"comma-separated, or 'all'. {POLICIES}")
    ap.add_argument("--cell-scheme", default="S3c",
                    choices=["S2", "S3", "S3c", "mt2net"],
                    help="P4_path_cell only: which sentence the cell chunk holds")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_arith")
    ap.add_argument("--rhb-question-types", nargs="*", default=[])
    ap.add_argument("--rhb-em-only", action="store_true")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0,
                    help="print this many random cases in full for eyeballing")
    ap.add_argument("--out-dir", default="results/hpc")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    pols = POLICIES if a.policy == "all" else tuple(
        p.strip() for p in a.policy.split(",") if p.strip())
    bad = set(pols) - set(POLICIES)
    if bad:
        ap.error(f"unknown policy {sorted(bad)}; pick from {POLICIES}")

    t0 = time.time()
    C = load_corpus(a)
    pop = C.queries[:a.max_queries] if a.max_queries else C.queries
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_text)} cells | "
          f"[pop] {len(pop)} queries | loaded in {time.time() - t0:.0f}s", flush=True)
    bud = Budget(a.embed_model)
    for p in pols:
        t1 = time.time()
        summ, _ = run(C, pop, a.dataset, p, bud, a.cell_scheme, a.out_dir,
                      a.verify, a.seed)
        print(f"\n[{p}] {json.dumps(summ)}  ({time.time() - t1:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
