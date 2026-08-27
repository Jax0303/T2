#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Split a corpus's queries by whether the GOLD cell's address is unique.

The repo predicts a corpus's gain from its collision rate, but collision has
always been counted over ALL cells. What decides a query is whether the address
of ITS gold cell is unique -- a corpus can be full of colliding cells nobody
asks about. This counts the share that matters and reads the committed runs
back through it, so the cross-corpus EM gap can be checked against a statistic
measurable with no LLM and no encoder.

Four groups, disjoint, worst wins:

  in-table        another cell in the SAME table writes the same address. The
                  header path failed at its one job; no encoder can separate
                  them, and neither can the reader once both are in context.
  twin DIFF value another TABLE writes the same address with a different value.
                  Ambiguous, and nothing in the header tree can fix it -- two
                  tables with the same header tree get the same structural term
                  (PREREG-2026-08-25-structural-discriminator.md, T0).
  twin same value another table writes the same address and the SAME value.
                  Harmless: either cell answers the question.
  clean           the address is unique corpus-wide.

    PYTHONPATH=.:scripts .venv/bin/python scripts/address_ambiguity.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GROUPS = ("clean", "twin same value", "twin DIFF value", "in-table")


def _norm(s) -> str:
    return re.sub(r"[\s,$%]", "", str(s)).strip().lower()


def label_queries(C) -> dict:
    """query_id -> group. The worst group any of the query's gold cells is in."""
    by_addr, idx, owners = defaultdict(list), {}, defaultdict(int)
    for n, ((rp, cp, _v), (t, i, j)) in enumerate(zip(C.cell_paths, C.cell_owner)):
        a = " > ".join(rp) + " | " + " > ".join(cp)
        by_addr[a].append(n)
        idx[(t, i, j)] = n
        owners[(t, a)] += 1
    out = {}
    for q in C.queries:
        rank = 0                                    # index into GROUPS
        for gc in q["gold_cells"]:
            n = idx.get(tuple(gc))
            if n is None:
                continue
            rp, cp, v = C.cell_paths[n]
            a = " > ".join(rp) + " | " + " > ".join(cp)
            tid = C.cell_owner[n][0]
            if owners[(tid, a)] > 1:
                rank = 3
                break
            twins = [m for m in by_addr[a] if C.cell_owner[m][0] != tid]
            if twins:
                rank = max(rank, 2 if any(_norm(C.cell_paths[m][2]) != _norm(v)
                                          for m in twins) else 1)
        out[q["query_id"]] = GROUPS[rank]
    return out


def read_run(path: str, label: dict, arm: str = "cell") -> dict:
    rows = defaultdict(lambda: {"n": 0, "em": 0, "osc": 0})
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = label.get(r["query_id"])
        if g is None or arm not in r:
            continue
        rows[g]["n"] += 1
        rows[g]["em"] += r[arm]["answer_em"]
        rows[g]["osc"] += r[arm]["osc"]
    return {g: {"n": v["n"], "em": round(v["em"] / v["n"], 4),
                "osc": round(v["osc"] / v["n"], 4)}
            for g, v in rows.items() if v["n"]}


CORPORA = {
    "hitab": ("results/wo_hitab_s2_512_records.jsonl",
              lambda a: __import__("corpus_dump_vs_cell").hitab_corpus(
                  a.data_dir, "dev", "hitab_dev_lookup_all")),
    "aitqa": ("results/wo_aitqa_s2_512_records.jsonl",
              lambda a: __import__("corpus_dump_vs_cell").aitqa_corpus()),
    # the header-column fix re-froze rhb_lookup_all at 231, so the run this reads
    # has to be the one produced on that population -- pairing the 243-query run
    # against 231 labels silently drops 28 queries and keeps 16 unlabelled
    "realhitbench": ("results/rhb_hdrfix_512_s2_records.jsonl",
                     lambda a: __import__("corpus_dump_vs_cell").realhitbench_corpus()),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--arm", default="cell")
    ap.add_argument("--out", default="results/address_ambiguity.json")
    args = ap.parse_args()
    out = {"metric": "EM and OSC by gold-address ambiguity, S2, budget 512",
           "arm": args.arm, "groups": {}, "runs": {}}
    print(f"{'corpus':14}{'group':18}{'n':>5}{'share':>8}{'EM':>8}{'OSC':>8}")
    for name, (run, build) in CORPORA.items():
        if not Path(run).exists():
            print(f"{name}: {run} missing, skipped")
            continue
        label = label_queries(build(args))
        rows = read_run(run, label, args.arm)
        tot = sum(v["n"] for v in rows.values())
        out["runs"][name] = run
        out["groups"][name] = {"total": tot, "by_group": rows}
        for g in GROUPS:
            if g in rows:
                v = rows[g]
                print(f"{name:14}{g:18}{v['n']:5}{v['n']/tot:8.1%}"
                      f"{v['em']:8.3f}{v['osc']:8.3f}")
    # How much of the cross-corpus EM gap is the MIX, and how much is the
    # per-group EM? Reweighting each corpus to HiTab's mix while keeping its own
    # group EMs separates the two, and the observed-EM column checks the
    # decomposition: it has to reproduce the run's headline number.
    ref = out["groups"].get("hitab", {}).get("by_group", {})
    ref_tot = out["groups"].get("hitab", {}).get("total", 0)
    if ref_tot:
        share = {g: ref[g]["n"] / ref_tot for g in ref}
        print(f"\n{'corpus':14}{'observed':>10}{'hitab mix':>11}{'mix worth':>11}")
        for name, v in out["groups"].items():
            rows, tot = v["by_group"], v["total"]
            obs = sum(r["n"] / tot * r["em"] for r in rows.values())
            cf = sum(share.get(g, 0) * rows[g]["em"] for g in rows)
            # a group HiTab has and this corpus does not contributes nothing
            v["observed_em"] = round(obs, 4)
            v["em_under_hitab_mix"] = round(cf, 4)
            v["mix_worth"] = round(cf - obs, 4)
            print(f"{name:14}{obs:10.3f}{cf:11.3f}{cf - obs:+11.3f}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
