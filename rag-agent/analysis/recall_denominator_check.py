#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Both recall denominators, side by side, for every arm measured so far."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("results/audit/recall_denominators.json")
RET = [Path("results/phase4/retrieval_294"),
       Path("results/phase4/retrieval_ext129")]


def from_retrieval(policy, pool):
    cell_hit = cell_n = q_hit = q_n = 0
    for rd in RET:
        f = rd / f"{policy}_{pool}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            d = json.loads(line)
            kept = set()
            for c in d["topk"]:
                if c["used"]:
                    for (i, j) in c["cells"]:
                        kept.add((c["table_id"], i, j))
            gold = [(str(g[0]), int(g[1]), int(g[2])) for g in d["gold_cells"]]
            if not gold:
                continue
            hits = [int(g in kept) for g in gold]
            cell_hit += sum(hits)
            cell_n += len(hits)
            q_hit += int(all(hits))
            q_n += 1
    return cell_hit, cell_n, q_hit, q_n


def from_taskd(f):
    cell_hit = cell_n = q_hit = q_n = 0
    for line in open(f):
        d = json.loads(line)
        cell_hit += d["n_gold_hit"]
        cell_n += d["n_gold"]
        q_hit += int(d["n_gold"] > 0 and d["n_gold_hit"] == d["n_gold"])
        q_n += 1
    return cell_hit, cell_n, q_hit, q_n


def row(name, t):
    ch, cn, qh, qn = t
    rc = round(ch / cn, 6) if cn else None
    rq = round(qh / qn, 6) if qn else None
    return {"arm": name, "cell_hit": ch, "cell_n": cn, "recall_cell": rc,
            "query_hit": qh, "query_n": qn, "recall_query": rq,
            "same": (rc == rq) if (cn and qn) else None}


def main() -> int:
    rows = []
    for pol in ("P1_fixed_512", "P4_path_cell"):
        for pool in ("hitab_lookup", "hitab_arith", "aitqa",
                     "rhb_fact", "rhb_num"):
            rows.append(row(f"phase4|{pol}|{pool}", from_retrieval(pol, pool)))
    for arm in ("P4_recon_row", "P4_recon", "P1_guessed_boundary"):
        for pool in ("hitab_lookup", "hitab_arith"):
            f = Path(f"results/audit/taskd/{arm}_{pool}.jsonl")
            if f.exists():
                rows.append(row(f"taskd|{arm}|{pool}", from_taskd(f)))
    f = Path("results/audit/rec_rp/records.jsonl")
    if f.exists():
        d = [json.loads(l) for l in open(f)]
        rows.append({"arm": "rec_rp|P4_recon_row|hitab_lookup",
                     "cell_hit": None, "cell_n": None, "recall_cell": None,
                     "query_hit": sum(x["gold_in_topk"] for x in d),
                     "query_n": len(d),
                     "recall_query": round(
                         sum(x["gold_in_topk"] for x in d) / len(d), 6),
                     "same": None,
                     "note": "file stored only the per-query all() flag"})
    f = Path("results/audit/unaligned_reader.jsonl")
    if f.exists():
        d = [json.loads(l) for l in open(f)]
        h, n = sum(x["gold_in_context"] for x in d), len(d)
        rows.append({"arm": "unaligned|P1_fixed_512|hitab_lookup_80",
                     "cell_hit": h, "cell_n": n, "recall_cell": round(h / n, 6),
                     "query_hit": h, "query_n": n,
                     "recall_query": round(h / n, 6), "same": True,
                     "note": "single gold cell per query"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    for r in rows:
        print(f"{r['arm']:42s} cell {str(r['recall_cell']):9s} "
              f"({r['cell_hit']}/{r['cell_n']})  query "
              f"{str(r['recall_query']):9s} ({r['query_hit']}/{r['query_n']})"
              f"  same={r['same']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
