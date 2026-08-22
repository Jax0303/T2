#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How often do two cells in the corpus get the SAME sentence? LLM-free.

The open question this answers: MultiHiertt and AIT-QA score low, and the
suspicion is RESIDUAL LABEL COLLISION -- cell sentences from different tables
land on identical text, so no similarity function can tell them apart and the
budget fills with the wrong table's cells.

Two counts per (dataset, scheme):

  sentence   the rendered index unit, VALUE INCLUDED. What the index holds.
  address    the same sentence with the value removed. What a QUERY can match:
             a question asks *for* the value, so it never contains it. Two cells
             with the same address are indistinguishable to the retriever no
             matter how good the encoder is -- this is the number that predicts
             the ceiling.

Reported as the share of CELLS sitting in a collision class (not the share of
classes): a class of 40 hurts 40 cells. ``cross_table`` restricts to classes
spanning more than one table, the case that actually breaks retrieval -- a
duplicate inside one table still points the reader at the right table.

Corpora come from ``corpus_dump_vs_cell`` unchanged, so these are the exact
strings the head-to-head runs index.

  PYTHONPATH=. python3 scripts/cell_sentence_collision.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_dump_vs_cell import (aitqa_corpus, hitab_corpus, multihiertt_corpus,
                                 realhitbench_corpus)
from point3_reconstruction_cost import cell_text
from rag_agent.runenv import run_env
from rag_agent.serialization.caption import caption_sentence


def render(C, scheme: str, with_value: bool) -> list[str]:
    """The corpus' cell sentences under ``scheme``; ``with_value=False`` drops
    the value, leaving the part a query can actually match."""
    out = []
    for n, (rp, cp, v) in enumerate(C.cell_paths):
        if scheme == "S3":
            tid = C.cell_owner[n][0]
            out.append(caption_sentence(C.title.get(tid, ""), rp, cp,
                                        v if with_value else None))
        else:
            out.append(cell_text(rp, cp, v if with_value else "", scheme))
    return out


def collisions(texts: list[str], owners: list) -> dict:
    by_text = defaultdict(list)
    for t, (tid, *_) in zip(texts, owners):
        by_text[t].append(tid)
    n = len(texts)
    dup_cells = sum(len(v) for v in by_text.values() if len(v) > 1)
    xt = {t: v for t, v in by_text.items() if len(set(v)) > 1}
    xt_cells = sum(len(v) for v in xt.values())
    sizes = Counter(len(v) for v in by_text.values())
    return {
        "n_cells": n,
        "distinct": len(by_text),
        "dup_cell_rate": round(dup_cells / n, 4) if n else 0.0,
        "cross_table_cell_rate": round(xt_cells / n, 4) if n else 0.0,
        "cross_table_classes": len(xt),
        "max_class": max(sizes) if sizes else 0,
        "max_class_tables": max((len(set(v)) for v in xt.values()), default=1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cell_sentence_collision.json")
    args = ap.parse_args()

    corpora = {
        "hitab": lambda: hitab_corpus(args.data_dir, args.split, ""),
        "multihiertt": lambda: multihiertt_corpus(args.mh_queries, args.seed),
        "aitqa": lambda: aitqa_corpus(),
        "realhitbench": lambda: realhitbench_corpus(),
    }
    out = {"experiment": "cell-sentence collision rate (no LLM, no encoder)",
           "env": run_env(args.seed, "none — no encoder in this diagnostic"), "datasets": {}}
    for name, load in corpora.items():
        C = load()
        d = {"tables": len(C.tids), "titled": round(
            sum(1 for t in C.tids if C.title.get(t)) / max(1, len(C.tids)), 4)}
        # A corpus where only SOME tables carry a title splits the title's effect
        # from everything else about a dataset. Across corpora the title is
        # confounded with domain, reader difficulty and sentence length; inside
        # one, it is not.
        has_title = [bool(C.title.get(t)) for t, _, _ in C.cell_owner]
        mixed = 0 < sum(has_title) < len(has_title)
        for scheme in ("flat", "S2", "S3"):
            d[scheme] = {
                "sentence": collisions(render(C, scheme, True), C.cell_owner),
                "address": collisions(render(C, scheme, False), C.cell_owner),
            }
            if mixed:
                addr = render(C, scheme, False)
                for lab, want in (("titled", True), ("untitled", False)):
                    idx = [k for k, t in enumerate(has_title) if t == want]
                    d[scheme][lab] = collisions([addr[k] for k in idx],
                                                [C.cell_owner[k] for k in idx])
        out["datasets"][name] = d
        print(f"[{name}] {d['tables']} tables, title {d['titled']:.1%}", flush=True)
        for scheme in ("flat", "S2", "S3"):
            a, s = d[scheme]["address"], d[scheme]["sentence"]
            if "titled" in d[scheme]:
                t_, u_ = d[scheme]["titled"], d[scheme]["untitled"]
                print(f"  {scheme:4s} titled {t_['dup_cell_rate']:.1%} "
                      f"(cross-table {t_['cross_table_cell_rate']:.1%})"
                      f"  |  untitled {u_['dup_cell_rate']:.1%} "
                      f"(cross-table {u_['cross_table_cell_rate']:.1%})", flush=True)
            print(f"  {scheme:4s} address dup {a['dup_cell_rate']:.1%} "
                  f"(cross-table {a['cross_table_cell_rate']:.1%}, "
                  f"worst class {a['max_class']} cells / {a['max_class_tables']} tables)"
                  f" | sentence dup {s['dup_cell_rate']:.1%}"
                  f" (cross-table {s['cross_table_cell_rate']:.1%})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
