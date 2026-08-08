#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The same four-arm comparison as baseline_comparison_llm.py, on MultiHiertt.

The HiTab run answered its own question badly: 256 of its 300 tables fit inside
the token budget whole, so "dump the table" needs no retrieval and ties us. That
result is about HiTab's table sizes, not about the method.

MultiHiertt does not have that escape hatch. A question is asked against a
DOCUMENT holding a median of 4 tables, so the whole evidence pool cannot be
pasted into the prompt and something has to choose what goes in. That is the
regime this method exists for, and it is the default here rather than a
44-question subgroup.

Arms, pool, budget and scoring are identical to the HiTab run so the two are
readable side by side. The only change is what the candidate pool is: every
parseable table of the question's own document, instead of one oracle table.

Population is table-only-evidence questions (``text_evidence`` empty). A method
that indexes cells cannot answer from a paragraph, and mixing those in would
score our inability to read prose as a serialization defect.

Run:
    PYTHONPATH=. .venv/bin/python scripts/baseline_comparison_multihiertt.py \
        --n 300 --budget 1024 --model openai:gpt-4o --resume
    PYTHONPATH=. .venv/bin/python scripts/baseline_comparison_multihiertt.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_dotenv():
    """Minimal .env loader (no dependency): rag-agent/.env then repo-root/.env."""
    here = Path(__file__).resolve().parent.parent
    for env in (here / ".env", here.parent / ".env"):
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

import numpy as np

from rag_agent.eval.metrics import hitab_exact_match_text
from rag_agent.generate.answerer import _DIRECT_SYS
from rag_agent.llm.factory import build_llm
from rag_agent.reconstruct import (guess_n_header_cols, guess_n_header_rows,
                                   parse_html_table, reconstruct_col_paths,
                                   reconstruct_row_paths)
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.runenv import run_env
from baseline_comparison_llm import ARMS, Budget, holm, mcnemar, rank

OURS = "cell_sent"


def load_population(n: int, seed: int):
    """Table-only questions, split into the two regimes, gold cells kept.

    ``lookup``    no program, exactly one evidence cell — find one number.
    ``aggregate`` a program over >= 2 evidence cells — find a SET, then compute.
    """
    from datasets import load_dataset
    rows = load_dataset("bevaya/MultiHiertt", split="train").to_list()

    pool = {"lookup": [], "aggregate": []}
    for row in rows:
        if row.get("text_evidence"):
            continue
        ev = row.get("table_evidence") or []
        if not ev:
            continue
        coords = []
        for e in ev:
            parts = e.split("-")
            if len(parts) != 3:
                coords = None
                break
            coords.append(tuple(int(x) for x in parts))
        if not coords:
            continue
        has_prog = bool((row.get("program") or "").strip())
        if not has_prog and len(coords) == 1:
            kind = "lookup"
        elif has_prog and len(coords) >= 2:
            kind = "aggregate"
        else:
            continue
        pool[kind].append({"uid": row["uid"], "question": row["question"],
                           "answer": row["answer"], "kind": kind,
                           "tables": row["tables"], "cells": coords})

    rng = random.Random(seed)
    for v in pool.values():
        rng.shuffle(v)
    n_agg = min(len(pool["aggregate"]), n // 2)
    out = pool["aggregate"][:n_agg] + pool["lookup"][: n - n_agg]
    rng.shuffle(out)
    return out, {k: len(v) for k, v in pool.items()}


def parse_doc(q):
    """Every parseable table of the question's document, with header paths.

    Returns ``None`` when the table holding the gold cells did not parse or the
    gold cells landed in a header region — those questions are unanswerable from
    this pool for EVERY arm, so scoring them would just add shared noise.
    """
    tabs = {}
    for t_idx, html in enumerate(q["tables"]):
        grid = parse_html_table(html)
        if len(grid) < 3 or len(grid[0]) < 2:
            continue
        nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=1), len(grid) - 1))
        nhc = max(1, min(guess_n_header_cols(grid, n_header_rows=nhr),
                         len(grid[0]) - 1))
        tabs[t_idx] = {"grid": grid, "nhr": nhr, "nhc": nhc,
                       "rows": reconstruct_row_paths(grid, nhr, nhc),
                       "cols": reconstruct_col_paths(grid, nhr, nhc)}
    for t_idx, r, c in q["cells"]:
        t = tabs.get(t_idx)
        if t is None or not (t["nhr"] <= r < len(t["grid"])
                             and t["nhc"] <= c < len(t["grid"][0])):
            return None
    return tabs


def doc_units(tabs):
    """The document rendered three ways, plus its markdown lines."""
    md, rows_u, flat_u, sent_u = [], [], [], []
    for t_idx in sorted(tabs):
        t = tabs[t_idx]
        grid, nhr, nhc = t["grid"], t["nhr"], t["nhc"]
        width = max(len(r) for r in grid)
        g = [[(x or "").strip() for x in r] + [""] * (width - len(r)) for r in grid]
        md.append(f"TABLE {t_idx}")
        md += ["| " + " | ".join(r) + " |" for r in g[:nhr]]
        md.append("|" + "---|" * width)
        md += ["| " + " | ".join(r) + " |" for r in g[nhr:]]
        for r in range(nhr, len(g)):
            rp = t["rows"][r - nhr] if (r - nhr) < len(t["rows"]) else []
            cells = []
            for c in range(nhc, width):
                v = g[r][c]
                if not v:
                    continue
                cp = t["cols"][c - nhc] if (c - nhc) < len(t["cols"]) else []
                leaf_r = rp[-1] if rp else ""
                leaf_c = cp[-1] if cp else ""
                lab = " ".join(x for x in (leaf_r, leaf_c) if x)
                flat_u.append(f"{lab}: {v}" if lab else v)
                path = " > ".join([*rp, *cp])
                sent_u.append(f"{path}: {v}" if path else v)
                cells.append(f"{leaf_c}: {v}" if leaf_c else v)
            if cells:
                head = " > ".join(rp)
                rows_u.append(f"{head} | " + " | ".join(cells) if head
                              else " | ".join(cells))
    return md, rows_u, flat_u, sent_u


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--model", default="openai:gpt-4o")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="results/baseline_comparison_multihiertt.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    rec_path = Path(str(Path(args.out).with_suffix("")) + "_records.jsonl")

    raw_pop, avail = load_population(args.n * 2, args.seed)
    pop, n_unparsed = [], 0
    for q in raw_pop:
        tabs = parse_doc(q)
        if tabs is None:
            n_unparsed += 1
            continue
        q["parsed"] = tabs
        pop.append(q)
        if len(pop) >= args.n:
            break
    kinds = {k: sum(1 for q in pop if q["kind"] == k) for k in ("lookup", "aggregate")}
    print(f"[pop] {len(pop)}문항 {kinds} | 파싱 실패로 제외 {n_unparsed} | 후보 {avail}",
          flush=True)

    enc = default_encoder(model_name=args.embed_model)
    bud = Budget(args.embed_model)

    def contexts(q):
        md, rows_u, flat_u, sent_u = doc_units(q["parsed"])
        qv = np.asarray(enc.encode([q["question"]])[0])
        kept_md, _ = bud.fill(md, args.budget)
        out = {"table_md": "\n".join(kept_md)}
        truncated = len(kept_md) < len(md)
        for arm, units in (("row_chunk", rows_u), ("flat_cell", flat_u),
                           ("cell_sent", sent_u)):
            kept, _ = bud.fill(rank(units, qv, enc), args.budget)
            out[arm] = "\n".join(kept)
        return out, truncated, len(q["parsed"])

    if args.dry_run:
        stats, n_tr, n_tabs = defaultdict(list), 0, []
        for q in pop[:40]:
            ctx, tr, nt = contexts(q)
            n_tr += tr
            n_tabs.append(nt)
            for a in ARMS:
                stats[a].append(bud.count(ctx[a]))
        print(json.dumps({
            "budget": args.budget, "sampled": min(40, len(pop)),
            "docs_whose_tables_did_not_fit": n_tr,
            "tables_per_doc_mean": round(float(np.mean(n_tabs)), 2),
            "context_tokens_mean": {a: round(float(np.mean(v)), 1)
                                    for a, v in stats.items()},
        }, indent=2, ensure_ascii=False))
        return 0

    done = {}
    if args.resume and rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            done[(r["uid"], r["arm"])] = r["correct"]
        print(f"[resume] 이미 채점된 (문항,방식) {len(done)}건", flush=True)

    llm = build_llm(args.model)
    rec_fh = open(rec_path, "a")
    n_trunc = 0
    try:
        for n, q in enumerate(pop, 1):
            if all((q["uid"], a) in done for a in ARMS):
                continue
            ctx, tr, n_tab = contexts(q)
            n_trunc += tr
            for arm in ARMS:
                if (q["uid"], arm) in done:
                    continue
                user = (f"CONTEXT:\n{ctx[arm]}\n\n"
                        f"QUESTION: {q['question']}\n\nAnswer:")
                pred = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=512)
                if not pred and llm.last_finish_reason == "length":
                    pred = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=1024)
                ok = bool(hitab_exact_match_text(pred, q["answer"]))
                done[(q["uid"], arm)] = ok
                rec_fh.write(json.dumps({
                    "uid": q["uid"], "arm": arm, "kind": q["kind"], "correct": ok,
                    "pred": pred[:120], "gold": q["answer"],
                    "ctx_tokens": bud.count(ctx[arm]),
                    "n_tables": n_tab, "md_truncated": tr}) + "\n")
                rec_fh.flush()
            if n % 20 == 0:
                cur = {a: f"{sum(v for (i, x), v in done.items() if x == a)}/"
                          f"{sum(1 for (i, x) in done if x == a)}" for a in ARMS}
                print(f"  {n}/{len(pop)}  {cur}", flush=True)
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"[중단] {str(e)[:160]} ... --resume 로 이어서 실행", flush=True)
    finally:
        rec_fh.close()

    kind_of = {q["uid"]: q["kind"] for q in pop}
    complete = [q["uid"] for q in pop if all((q["uid"], a) in done for a in ARMS)]
    spent = defaultdict(list)
    if rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            spent[r["arm"]].append(r.get("ctx_tokens"))

    def block(ids):
        if not ids:
            return {}
        acc = {a: round(sum(done[(i, a)] for i in ids) / len(ids), 4) for a in ARMS}
        n_correct = {a: sum(done[(i, a)] for i in ids) for a in ARMS}
        tests, raw_p = {}, []
        for a, b_ in combinations(ARMS, 2):
            b = sum(1 for i in ids if done[(i, a)] and not done[(i, b_)])
            c = sum(1 for i in ids if done[(i, b_)] and not done[(i, a)])
            lab, p = f"{a}_vs_{b_}", mcnemar(
                sum(1 for i in ids if done[(i, a)] and not done[(i, b_)]),
                sum(1 for i in ids if done[(i, b_)] and not done[(i, a)]))
            tests[lab] = {"only_" + a: b, "only_" + b_: c, "p_mcnemar": round(p, 6)}
            raw_p.append((lab, p))
        for lab, adj in holm(raw_p).items():
            tests[lab]["p_holm"] = adj
        return {"n_questions": len(ids), "n_correct": n_correct, "accuracy": acc,
                "delta_ours_minus": {a: round(acc[OURS] - acc[a], 4)
                                     for a in ARMS if a != OURS},
                "pairwise": tests}

    out = {
        "env": env,
        "leg": "cell-sentence RAG vs whole-document-markdown and row-chunk table "
               "RAG on MultiHiertt, equal token budget, pool = every table of the "
               "question's document",
        "dataset": "multihiertt_train_table_only_evidence",
        "solver": args.model,
        "scorer": "hitab_exact_match_text",
        "budget_tokens": args.budget,
        "arms": {"table_md": "every table of the document as markdown",
                 "row_chunk": "one chunk per data row, ranked",
                 "flat_cell": "cell sentence, leaf labels only",
                 "cell_sent": "cell sentence, full row-path > col-path (ours)"},
        "population": {"n_sampled": len(pop), "n_complete_all_arms": len(complete),
                       "by_kind": kinds, "n_dropped_unparsed": n_unparsed,
                       "available": avail},
        "n_docs_not_fitting_budget": n_trunc,
        "context_tokens_mean_actual": {
            a: (round(float(np.mean([x for x in v if x is not None])), 1) if v else None)
            for a, v in spent.items()},
        "overall": block(complete),
        "lookup": block([i for i in complete if kind_of[i] == "lookup"]),
        "aggregate": block([i for i in complete if kind_of[i] == "aggregate"]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("overall", "lookup", "aggregate")}, indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
