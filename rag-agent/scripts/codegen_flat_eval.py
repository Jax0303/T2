#!/usr/bin/env python3
"""Flat-table codegen eval: does the grounded self-repair method degrade on
SIMPLE (flat) tables vs naive codegen?  Milestone 1 — "no regression on flat".

A flat WikiTableQuestions table is loaded as a depth-1 OriginalTable
(top_paths = [[col]], left_paths = [[first-col value]]), so the SAME machinery
from method_grounded.py (TracedTable, grounding-trace self-repair, prompts)
runs unchanged. The header-path component is a no-op on flat (paths are length
1); what is tested here is whether the *grounding + self-repair* part is at
least non-harmful on simple tables.

Both modes run on the SAME sampled questions (paired) so naive vs grounded is
a clean within-question comparison. LLM = Groq (CPU-friendly; local 7B is not).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.stores.original_store import OriginalTable          # noqa: E402
from rag_agent.eval.metrics import numeric_match, exact_match      # noqa: E402
from rag_agent.llm.factory import build_llm                        # noqa: E402
import method_grounded as mg                                        # noqa: E402

SEED = 42


def complete_retry(llm, sys_p, user, max_tokens, tries=6):
    """Groq call with exponential backoff on transient errors (rate limits)."""
    delay = 4.0
    last = None
    for _ in range(tries):
        try:
            return llm.complete(sys_p, user, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise last


def flat_table(tid, title, header, rows):
    """A flat table as a degenerate (depth-1) hierarchical OriginalTable.

    Row binding (lever 2): instead of keying each row by its first column only,
    the row path is ALL non-empty cell values of the row, so a row_header /
    row_filter can match an entity living in any column (WikiSQL WHERE values,
    WTQ key entities that are not in column 0).
    """
    data = [list(r) for r in rows]
    top_paths = [[str(h)] for h in header]
    left_paths = []
    for r in rows:
        seen, path = set(), []
        for v in r:
            s = str(v).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                path.append(s)
        left_paths.append(path or [""])
    return OriginalTable(
        table_id=tid, title=title or "", data=data,
        top_paths=top_paths, left_paths=left_paths,
        top_paths_by_col={i: p for i, p in enumerate(top_paths)},
        left_paths_by_row={i: p for i, p in enumerate(left_paths)},
    )


# ---- WikiSQL: load from the GitHub release + execute SQL for the gold answer ---
_AGG = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]


def _isnum(v):
    try:
        float(str(v).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _num(v):
    return float(str(v).replace(",", ""))


def execute_wikisql(table, sql):
    """Execute a WikiSQL query over its table; return the gold answer list."""
    rows = table["rows"]
    sel, agg, conds = sql["sel"], sql["agg"], sql["conds"]

    def match(row):
        for ci, op, val in conds:
            c = row[ci]
            if op == 0:  # =
                if str(c).strip().lower() != str(val).strip().lower():
                    return False
            elif op == 1:  # >
                if not (_isnum(c) and _isnum(val) and _num(c) > _num(val)):
                    return False
            elif op == 2:  # <
                if not (_isnum(c) and _isnum(val) and _num(c) < _num(val)):
                    return False
        return True

    sel_vals = [row[sel] for row in rows if match(row)]
    if agg == 0:
        return sel_vals
    if agg == 3:  # COUNT
        return [len(sel_vals)]
    nums = [_num(v) for v in sel_vals if _isnum(v)]
    if not nums:
        return []
    if agg == 1:
        return [max(nums)]
    if agg == 2:
        return [min(nums)]
    if agg == 4:
        return [sum(nums)]
    if agg == 5:
        return [sum(nums) / len(nums)]
    return sel_vals


def load_wikisql(n, seed, wikisql_dir="/tmp/WikiSQL/data", split="dev"):
    import json as _json
    base = Path(wikisql_dir)
    tables = {}
    with open(base / f"{split}.tables.jsonl") as f:
        for line in f:
            t = _json.loads(line)
            tables[t["id"]] = t
    examples = []
    with open(base / f"{split}.jsonl") as f:
        for line in f:
            examples.append(_json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(examples)
    out = []
    for e in examples:
        t = tables.get(e["table_id"])
        if not t or not t.get("rows"):
            continue
        gold = execute_wikisql(t, e["sql"])
        if not gold:
            continue
        out.append({
            "id": e["table_id"],
            "question": e["question"],
            "answers": [str(g) for g in gold],
            "table": {"header": t["header"], "rows": t["rows"]},
            "agg": _AGG[e["sql"]["agg"]] or "select",
        })
        if len(out) >= n:
            break
    return out


def load_wtq(n, seed):
    from datasets import load_dataset
    ds = load_dataset("lighteval/wikitablequestions", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        t = ex.get("table") or {}
        if not t.get("header") or not t.get("rows") or not ex.get("answers"):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def _quality(result, err, trace):
    """Higher is better: prefer no-error > non-empty > fewer grounding flags.
    Used by the over-correction safeguard so a repair is only adopted when it
    does not make the answer worse."""
    flags = sum(1 for t in trace if t.get("flag"))
    nonempty = result is not None and result != "" and result != []
    return (not bool(err), bool(nonempty), -flags)


def run_one(ex, mode, llm, repairs, max_tokens, idx):
    q = ex["question"]
    gold = ex["answers"]
    ot = flat_table(ex.get("id", f"wtq-{idx}"), "", ex["table"]["header"], ex["table"]["rows"])
    tt = mg.TracedTable(ot)
    api = {"cell": tt.cell, "col_values": tt.col_values, "row_values": tt.row_values,
           "list_rows": tt.list_rows, "list_cols": tt.list_cols}
    grounded = mode == "grounded"
    sys_p = mg.GROUNDED_SYS if grounded else mg.NAIVE_SYS
    user = mg.build_user(ot.title, q, tt, grounded=grounded)
    try:
        code = mg.strip_code(complete_retry(llm, sys_p, user, max_tokens))
    except Exception as e:  # noqa: BLE001
        return {"query": q, "gold": gold, "pred": "", "correct": False,
                "err": f"LLM:{type(e).__name__}", "n_repair": 0, "code": ""}
    tt.trace = []
    result, err = mg.run_code(code, api)
    n_repair = 0
    # over-correction safeguard: keep the best result seen across repairs
    best = (result, err, code, _quality(result, err, tt.trace))
    if grounded:
        while n_repair < repairs and mg.needs_repair(tt.trace, result, err):
            fb = mg.trace_feedback(code, tt.trace, result, err)
            try:
                code = mg.strip_code(complete_retry(llm, sys_p, user + "\n\n" + fb, max_tokens))
            except Exception as e:  # noqa: BLE001
                break
            tt.trace = []
            result, err = mg.run_code(code, api)
            n_repair += 1
            qy = _quality(result, err, tt.trace)
            if qy > best[3]:
                best = (result, err, code, qy)
        result, err, code = best[0], best[1], best[2]   # adopt repair only if better
    pred = "" if result is None else str(result)
    ok = numeric_match(pred, gold) or exact_match(pred, gold)
    return {"query": q, "gold": gold, "pred": pred, "correct": bool(ok),
            "err": err, "n_repair": n_repair, "code": code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wtq", "wikisql"], default="wtq")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--llm", default="groq:llama-3.3-70b-versatile")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--sleep", type=float, default=0.4, help="pause between LLM calls (rate limit)")
    ap.add_argument("--out", default="results/codegen/flat_wtq.json")
    args = ap.parse_args()

    exs = load_wikisql(args.n, SEED) if args.dataset == "wikisql" else load_wtq(args.n, SEED)
    llm = build_llm(args.llm)
    print(f"WTQ flat codegen | n={len(exs)} | llm={args.llm} | repairs={args.repairs}", flush=True)

    out = {"config": vars(args), "n": len(exs), "modes": {}}
    for mode in ["naive", "grounded"]:
        rows = []
        t0 = time.time()
        for i, ex in enumerate(exs, 1):
            rows.append(run_one(ex, mode, llm, args.repairs, args.max_tokens, i))
            time.sleep(args.sleep)
            if i % 10 == 0 or i == len(exs):
                acc = sum(r["correct"] for r in rows) / len(rows)
                print(f"  [{mode}] {i}/{len(exs)} acc={acc:.3f} {time.time()-t0:.0f}s", flush=True)
        acc = sum(r["correct"] for r in rows) / len(rows)
        out["modes"][mode] = {"acc": acc, "rows": rows}
        print(f"== {mode}: acc={acc:.3f} ==", flush=True)

    na, gr = out["modes"]["naive"]["acc"], out["modes"]["grounded"]["acc"]
    # paired: per-question correctness
    nc = [r["correct"] for r in out["modes"]["naive"]["rows"]]
    gc = [r["correct"] for r in out["modes"]["grounded"]["rows"]]
    flipped_up = sum(1 for a, b in zip(nc, gc) if (not a) and b)
    flipped_dn = sum(1 for a, b in zip(nc, gc) if a and (not b))
    out["summary"] = {"naive_acc": na, "grounded_acc": gr, "delta": gr - na,
                      "rescued": flipped_up, "broke": flipped_dn}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nNAIVE={na:.3f}  GROUNDED={gr:.3f}  Δ={gr-na:+.3f}  "
          f"(rescued {flipped_up}, broke {flipped_dn})  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
