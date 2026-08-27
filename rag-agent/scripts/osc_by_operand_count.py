"""Does the cell arm's OSC advantage survive as the operand set grows?

The thesis sells cell-level addressing on *set* completeness: a query needing
several operand cells is scored all-or-nothing, and cells-as-documents is
supposed to be what assembles the set. That story has never been checked
against ``m`` -- every headline number is an average over a population whose
mean is 2.7 but whose advantage could live anywhere inside it.

Splits every committed dump-vs-cell record file by gold-operand count and
prints OSC per bucket. Retrieval only; no reader, no API, nothing to re-run.
"""
from __future__ import annotations

import collections
import json
import sys

BUCKETS = ("m=1", "m=2", "m>=3")


def bucket(m: int) -> str:
    return "m=1" if m == 1 else ("m=2" if m == 2 else "m>=3")


def split(path: str, arms=("dump", "cell")):
    """``{bucket: (n, *per-arm mean OSC)}``, or None if the file predates ``m``."""
    rows = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("m") is None:
                return None
            rows[bucket(r["m"])].append(r)
    return {
        k: (len(rs), *[sum(r[a]["osc"] for r in rs) / len(rs) for a in arms])
        for k, rs in rows.items()
    }


def report(label: str, paths: dict) -> dict:
    print(f"\n=== {label} ===")
    print(f"{'budget':>7} | " + " | ".join(f"{b:^24}" for b in BUCKETS))
    print(f"{'':>7} | " + " | ".join(f"{'n':>4} {'dump':>5} {'cell':>5} {'delta':>6}" for _ in BUCKETS))
    out = {}
    for budget, path in sorted(paths.items()):
        got = split(path)
        if got is None:
            print(f"{budget:>7} | (no m field -- file predates the field)")
            continue
        cells = []
        for b in BUCKETS:
            if b not in got:
                cells.append(f"{'--':>4} {'':>5} {'':>5} {'':>6}")
                continue
            n, d, c = got[b]
            cells.append(f"{n:>4} {d:>5.3f} {c:>5.3f} {c - d:>+6.3f}")
        print(f"{budget:>7} | " + " | ".join(cells))
        out[budget] = {b: {"n": got[b][0], "dump": got[b][1], "cell": got[b][2],
                           "delta": got[b][2] - got[b][1]} for b in got}
    return out


def main() -> int:
    families = {
        "HiTab hitab_dev_corpus_arith, dense": {
            b: f"results/corpus_dump_vs_cell_dense_{b}_records.jsonl"
            for b in (128, 256, 512, 1024, 2048, 4096)},
        "HiTab hitab_dev_corpus_arith, bm25": {
            b: f"results/corpus_dump_vs_cell_bm25_{b}_records.jsonl"
            for b in (128, 256, 512, 1024, 2048, 4096)},
        "MultiHiertt n=400, bm25": {
            b: f"results/corpus_dump_vs_cell_multihiertt_bm25_{b}_records.jsonl"
            for b in (128, 256, 512, 1024, 2048)},
    }
    result = {label: report(label, paths) for label, paths in families.items()}

    # The all-or-nothing tax: cells the arm did retrieve, minus sets it completed.
    print("\n=== where the m>=3 loss comes from (dense, budget 2048) ===")
    recs = [json.loads(l) for l in
            open("results/corpus_dump_vs_cell_dense_2048_records.jsonl")]
    hard = [r for r in recs if r["m"] >= 3]
    for arm in ("dump", "cell", "cascade"):
        pc = sum(r[arm]["per_cell"] for r in hard) / len(hard)
        osc = sum(r[arm]["osc"] for r in hard) / len(hard)
        print(f"  {arm:9s} per_cell {pc:.3f}   osc {osc:.3f}   all-or-nothing tax {pc - osc:.3f}")
        result.setdefault("_tax_dense_2048_m3", {})[arm] = {
            "n": len(hard), "per_cell": pc, "osc": osc, "tax": pc - osc}

    out = "results/osc_by_operand_count.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
