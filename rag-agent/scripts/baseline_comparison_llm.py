#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cell-sentence RAG vs the table-RAG baselines people actually use.

Every answer-accuracy win this repo holds is against `flat` — our OWN flattened
cell serialization. That answers "does the header path help?", not "is this
better than existing table RAG?", which is the question the thesis has to
survive. So the two baselines a practitioner would actually reach for are arms
here:

  table_md   the whole table dumped as markdown, one block. The default move,
             and the strongest baseline when the table fits in the budget.
  row_chunk  one chunk per data row, ranked against the question. Standard
             practice when the table does not fit.
  flat_cell  cell sentences carrying only the LEAF row/col labels. Our own
             baseline, kept so the new numbers join up with the old ones.
  cell_sent  cell sentences carrying the FULL row-path > col-path. Ours.

FAIRNESS. All four arms get the same token budget, counted with one tokenizer,
and the same prompt frame -- only the contents of CONTEXT differ. Without this
the comparison is unreadable: a whole-table dump spends several thousand tokens
where top-k cell sentences spend a few hundred, so any win could be bought
rather than earned. Budget is filled greedily in rank order; ``table_md`` drops
trailing DATA rows (never headers) when the table overflows, and how often that
happens is reported, because that is the regime where a dump stops being viable.

The table gate is oracle for every arm (HiTab table recall@20 = 1.00 already),
so this isolates what happens INSIDE the right table.

Reported by question type, because the two regimes differ:
  lookup     one gold operand cell -- find one number
  aggregate  >=2 gold operand cells -- find a SET, then compute

Run:
    PYTHONPATH=. .venv/bin/python scripts/baseline_comparison_llm.py \
        --n 300 --budget 1024 --model openai:gpt-4o --resume
    PYTHONPATH=. .venv/bin/python scripts/baseline_comparison_llm.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
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

from rag_agent.bench.hitab import load_queries
from rag_agent.eval.metrics import hitab_exact_match_text
from rag_agent.generate.answerer import _DIRECT_SYS
from rag_agent.llm.factory import build_llm
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.runenv import run_env
from point3_reconstruction_cost import build_table_paths, cell_text

ARMS = ("table_md", "row_chunk", "flat_cell", "cell_sent")
OURS = "cell_sent"
ARITH = {"sum", "diff", "div", "average", "range", "opposite", "count", "counta"}


class Budget:
    """One tokenizer for all four arms.

    It is not the solver's tokenizer and does not need to be: fairness needs a
    common ruler, not the true one. Using the encoder's tokenizer keeps the
    ruler local, deterministic and already downloaded.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from transformers import AutoTokenizer
        # counting only, never encoding: raise the limit so long contexts do not
        # emit the 512-token truncation warning on every call
        self.tok = AutoTokenizer.from_pretrained(model_name, model_max_length=10**6)

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def fill(self, units, limit: int):
        """Greedily take units in rank order while they fit. Returns (kept, tokens)."""
        kept, used = [], 0
        for u in units:
            n = self.count(u)
            if used + n > limit:
                continue          # a long unit must not block every shorter one
            kept.append(u)
            used += n
        return kept, used


def markdown_table(raw, n_head_rows: int) -> list[str]:
    """The raw grid as markdown lines: a header block, then one line per data row."""
    grid = [[str(x) if x is not None else "" for x in row] for row in raw["texts"]]
    if not grid:
        return []
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    head = grid[:n_head_rows] or [grid[0]]
    lines = ["| " + " | ".join(r) + " |" for r in head]
    lines.append("|" + "---|" * width)
    lines += ["| " + " | ".join(r) + " |" for r in grid[n_head_rows:]]
    return lines


def row_chunks(bt, pt) -> list[str]:
    """One text per data row: its row path, then each column's label and value."""
    out = []
    for i in range(pt["n_r"]):
        rp = " > ".join(pt["gold_rp"][i])
        cells = []
        for j in range(pt["n_c"]):
            lab = pt["gold_cp"][j][-1] if pt["gold_cp"][j] else ""
            cells.append(f"{lab}: {bt.data[i][j]}")
        out.append(f"{rp} | " + " | ".join(cells) if rp else " | ".join(cells))
    return out


def cell_units(bt, pt, scheme: str) -> list[str]:
    return [cell_text(pt["gold_rp"][i], pt["gold_cp"][j], bt.data[i][j], scheme)
            for i in range(pt["n_r"]) for j in range(pt["n_c"])]


def rank(units, qv, enc, top: int = 400):
    """Rank units against the question by cosine, best first."""
    if not units:
        return []
    vecs = np.asarray(enc.encode(units))
    order = np.argsort(-(vecs @ qv))[:top]
    return [units[o] for o in order]


def mcnemar(b: int, c: int) -> float:
    from scipy.stats import binomtest
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def holm(pairs):
    """Holm-Bonferroni over (label, p). Returns {label: adjusted p}."""
    ordered = sorted(pairs, key=lambda t: t[1])
    m, out, running = len(ordered), {}, 0.0
    for i, (lab, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (m - i)))
        out[lab] = round(running, 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=1024, help="context tokens per arm")
    ap.add_argument("--model", default="openai:gpt-4o")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="results/baseline_comparison_llm.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="population + token stats only, no LLM calls")
    args = ap.parse_args()
    env = run_env(args.seed, args.embed_model)
    rec_path = Path(str(Path(args.out).with_suffix("")) + "_records.jsonl")

    queries, tables = load_queries(args.data_dir, args.split)
    raw_dir = Path(args.data_dir) / "data/tables/raw"
    paths, raws = {}, {}
    for tid, bt in tables.items():
        p = raw_dir / f"{tid}.json"
        if not p.exists():
            continue
        try:
            raw = json.load(open(p))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is not None:
            paths[tid], raws[tid] = pt, raw

    pop = []
    for q in queries:
        pt = paths.get(q.gold_table_id)
        if pt is None:
            continue
        ops = {(o.row, o.col) for o in q.gold_operands}
        if not ops or any(not (0 <= r < pt["n_r"] and 0 <= c < pt["n_c"])
                          for r, c in ops):
            continue
        q.kind = "lookup" if len(ops) == 1 else "aggregate"
        pop.append(q)

    # stratified: aggregate questions are the scarce, hard regime (and the one
    # this method is built for), so take as many as exist up to half the sample
    # instead of letting a proportional draw leave the subgroup underpowered
    rng = random.Random(args.seed)
    by_kind = {k: [q for q in pop if q.kind == k] for k in ("lookup", "aggregate")}
    for v in by_kind.values():
        rng.shuffle(v)
    n_agg = min(len(by_kind["aggregate"]), args.n // 2)
    pop = by_kind["aggregate"][:n_agg] + by_kind["lookup"][: args.n - n_agg]
    rng.shuffle(pop)
    kinds = {k: sum(1 for q in pop if q.kind == k) for k in ("lookup", "aggregate")}
    print(f"[pop] {len(pop)} queries  {kinds} "
          f"(available: { {k: len(v) for k, v in by_kind.items()} })", flush=True)

    enc = default_encoder(model_name=args.embed_model)
    bud = Budget(args.embed_model)

    def contexts(q):
        """The four CONTEXT blocks for one question, each within the budget."""
        tid = q.gold_table_id
        bt, pt, raw = tables[tid], paths[tid], raws[tid]
        qv = np.asarray(enc.encode([q.question])[0])
        n_head = max(1, len(raw["texts"]) - pt["n_r"])

        md = markdown_table(raw, n_head)
        head, body = md[: n_head + 1], md[n_head + 1:]
        head_tok = sum(bud.count(x) for x in head)
        kept_body, _ = bud.fill(body, max(0, args.budget - head_tok))
        out = {"table_md": "\n".join(head + kept_body)}
        truncated = len(kept_body) < len(body)

        for arm, units in (("row_chunk", row_chunks(bt, pt)),
                           ("flat_cell", cell_units(bt, pt, "flat")),
                           ("cell_sent", cell_units(bt, pt, "S2"))):
            kept, _ = bud.fill(rank(units, qv, enc), args.budget)
            out[arm] = "\n".join(kept)
        return out, truncated, len(md) - n_head - 1

    if args.dry_run:
        stats = {a: [] for a in ARMS}
        n_trunc = 0
        for q in pop[:60]:
            ctx, tr, _ = contexts(q)
            n_trunc += tr
            for a in ARMS:
                stats[a].append(bud.count(ctx[a]))
        print(json.dumps({
            "budget": args.budget,
            "sampled": min(60, len(pop)),
            "table_md_truncated": n_trunc,
            "context_tokens_mean": {a: round(float(np.mean(v)), 1)
                                    for a, v in stats.items()},
            "context_tokens_max": {a: int(max(v)) for a, v in stats.items()},
        }, indent=2))
        return 0

    done = {}
    if args.resume and rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            done[(r["query_id"], r["arm"])] = r["correct"]
        print(f"[resume] {len(done)} (query,arm) results recorded", flush=True)

    llm = build_llm(args.model)
    rec_fh = open(rec_path, "a")
    n_trunc = 0
    try:
        for n, q in enumerate(pop, 1):
            if all((q.query_id, a) in done for a in ARMS):
                continue
            ctx, tr, _ = contexts(q)
            n_trunc += tr
            for arm in ARMS:
                if (q.query_id, arm) in done:
                    continue
                user = (f"CONTEXT:\n{ctx[arm]}\n\n"
                        f"QUESTION: {q.question}\n\nAnswer:")
                raw_out = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=512)
                if not raw_out and llm.last_finish_reason == "length":
                    raw_out = llm.complete(system=_DIRECT_SYS, user=user,
                                           max_tokens=1024)
                ok = bool(hitab_exact_match_text(raw_out, q.answer))
                done[(q.query_id, arm)] = ok
                rec_fh.write(json.dumps({
                    "query_id": q.query_id, "arm": arm, "kind": q.kind,
                    "correct": ok, "pred": raw_out[:120],
                    # what the arm actually spent, so "equal budget" is a
                    # reported fact rather than a claim about the cap
                    "ctx_tokens": bud.count(ctx[arm]),
                    "md_truncated": tr}) + "\n")
                rec_fh.flush()
            if n % 20 == 0:
                cur = {a: f"{sum(v for (i, x), v in done.items() if x == a)}/"
                          f"{sum(1 for (i, x) in done if x == a)}" for a in ARMS}
                print(f"  {n}/{len(pop)}  {cur}", flush=True)
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"[stopped] {str(e)[:160]} ... rerun with --resume", flush=True)
    finally:
        rec_fh.close()

    kind_of = {q.query_id: q.kind for q in pop}
    complete = [q.query_id for q in pop if all((q.query_id, a) in done for a in ARMS)]

    spent = {a: [] for a in ARMS}
    if rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            if "ctx_tokens" in r:
                spent[r["arm"]].append(r["ctx_tokens"])

    def block(ids):
        if not ids:
            return {}
        acc = {a: round(sum(done[(i, a)] for i in ids) / len(ids), 4) for a in ARMS}
        tests, raw_p = {}, []
        for a, b_ in combinations(ARMS, 2):
            b = sum(1 for i in ids if done[(i, a)] and not done[(i, b_)])
            c = sum(1 for i in ids if done[(i, b_)] and not done[(i, a)])
            lab = f"{a}_vs_{b_}"
            p = mcnemar(b, c)
            tests[lab] = {"only_" + a: b, "only_" + b_: c, "p_mcnemar": round(p, 6)}
            raw_p.append((lab, p))
        for lab, adj in holm(raw_p).items():
            tests[lab]["p_holm"] = adj
        return {"n": len(ids), "accuracy": acc,
                "delta_ours_minus": {a: round(acc[OURS] - acc[a], 4)
                                     for a in ARMS if a != OURS},
                "pairwise": tests}

    out = {
        "env": env,
        "leg": "cell-sentence RAG vs whole-table-markdown and row-chunk table RAG, "
               "equal token budget, oracle table gate, official HiTab scorer",
        "dataset": f"hitab_{args.split}",
        "solver": args.model,
        "scorer": "hitab_exact_match_text (official scorer; free-text answers are "
                  "split into a value list when gold holds several values, which "
                  "is the shape HiTab's own eval receives -- no tolerance added)",
        "budget_tokens": args.budget,
        "budget_tokenizer": args.embed_model,
        "arms": {"table_md": "whole table as markdown, one block",
                 "row_chunk": "one chunk per data row, ranked",
                 "flat_cell": "cell sentence, leaf labels only (our old baseline)",
                 "cell_sent": "cell sentence, full row-path > col-path (ours)"},
        "population": {"n_sampled": len(pop), "n_complete_all_arms": len(complete),
                       "by_kind": kinds},
        "n_table_md_truncated": n_trunc,
        "context_tokens_mean_actual": {a: (round(float(np.mean(v)), 1) if v else None)
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
