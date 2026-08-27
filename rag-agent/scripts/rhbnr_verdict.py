# SPDX-License-Identifier: MIT
"""Judge PREREG-2026-08-27-rhb-nr-large-tables against the runs it predicted.

The prereg commits P1..P8 as numbers before the run. This reads the records back
and says which held, so the verdict is arithmetic rather than narration. It
prints nothing the prereg did not already ask for.

Run:
    PYTHONPATH=.:scripts python3 scripts/rhbnr_verdict.py \
        --a results/rhbnr_s3c_512_records.jsonl \
        --b results/rhbnr_tgpt_512_records.jsonl \
        --out results/rhbnr_verdict.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_comparison_llm import mcnemar

# The prereg's own buckets (§3). "small" is where handing the table over should
# win, "large" is where retrieval should.
BUCKETS = (("<=512", 0, 512), ("513-2048", 513, 2048), (">2048", 2049, 10 ** 9))


def load(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def em(recs: list, arm: str) -> float | None:
    v = [r[arm]["answer_em"] for r in recs if arm in r and r[arm].get("answer_em") is not None]
    return round(sum(v) / len(v), 4) if v else None


def paired(recs: list, a: str, b: str) -> dict:
    """Ours-minus-theirs on the SAME queries, with the sign that survives McNemar."""
    pairs = [(r[a]["answer_em"], r[b]["answer_em"]) for r in recs
             if a in r and b in r
             and r[a].get("answer_em") is not None and r[b].get("answer_em") is not None]
    win = sum(1 for x, y in pairs if x and not y)
    los = sum(1 for x, y in pairs if y and not x)
    d = (sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / len(pairs) if pairs else None
    return {"n": len(pairs), "delta": round(d, 4) if d is not None else None,
            "win": win, "loss": los, "p": mcnemar(win, los) if pairs else None}


def by_bucket(recs: list, a: str, b: str) -> dict:
    out = {}
    for name, lo, hi in BUCKETS:
        sub = [r for r in recs if lo <= r.get("gold_table_tokens", -1) <= hi]
        out[name] = {**paired(sub, a, b), a: em(sub, a), b: em(sub, b)}
    return out


def judge(recs: list, label: str) -> dict:
    """P1..P5 for one reader. P6/P8 need a second file and are done by main."""
    buckets = by_bucket(recs, "cell", "goldtable")
    big, small = buckets[">2048"], buckets["<=512"]
    p1 = big["delta"] is not None and big["delta"] >= 0.05
    p2 = small["delta"] is not None and -small["delta"] >= 0.03
    p5 = paired(recs, "cell", "flat")
    cell_em = em(recs, "cell")
    return {
        "reader": label, "n": len(recs),
        "em": {a: em(recs, a) for a in ("cell", "goldtable", "flat")},
        "by_gold_table_size": buckets,
        "P1_cell_beats_goldtable_on_big_tables": {
            "target": ">= +.05", "got": big["delta"], "held": bool(p1)},
        "P2_goldtable_beats_cell_on_small_tables": {
            "target": ">= +.03", "got": (None if small["delta"] is None
                                         else round(-small["delta"], 4)),
            "held": bool(p2)},
        "P3_signs_are_opposite": {
            "held": bool(big["delta"] is not None and small["delta"] is not None
                         and big["delta"] > 0 > small["delta"])},
        "P4_full_population_cell_em": {
            "target": ".03 - .08, and below the .0909 of the 55-query subset",
            "got": cell_em,
            "held": bool(cell_em is not None and 0.03 <= cell_em <= 0.08),
            "below_old_subset": bool(cell_em is not None and cell_em < 0.0909)},
        "P5_cell_beats_flat": {"target": ">= +.03", "got": p5["delta"],
                               "held": bool(p5["delta"] is not None
                                            and p5["delta"] >= 0.03),
                               "paired": p5},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="records.jsonl of reader A (Qwen2.5-7B)")
    ap.add_argument("--b", default="", help="records.jsonl of reader B (TableGPT2-7B)")
    ap.add_argument("--label-a", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--label-b", default="TableGPT2-7B")
    ap.add_argument("--out", default="results/rhbnr_verdict.json")
    args = ap.parse_args()

    out = {"prereg": "PREREG-2026-08-27-rhb-nr-large-tables.md",
           "readers": [judge(load(args.a), args.label_a)]}
    if args.b:
        rb = load(args.b)
        out["readers"].append(judge(rb, args.label_b))
        # P6: flat reads no template, so it must not move between the two runs.
        # Here the arms share one run per reader, so this compares READERS -- it
        # is expected to move, and is reported as context rather than as a gate.
        out["P6_flat_control"] = {
            "note": "flat differs across READERS by design; the within-run flat "
                    "control is automatic because all arms share one run",
            "flat_em": {args.label_a: em(load(args.a), "flat"),
                        args.label_b: em(rb, "flat")}}
        held = [r["P1_cell_beats_goldtable_on_big_tables"]["held"]
                for r in out["readers"]]
        out["P8_P1_holds_under_both_readers"] = {
            "held": bool(all(held)),
            "verdict": ("the claim is about context length"
                        if all(held) else
                        "scope-limited: report as 'only when the reader cannot "
                        "read the table anyway' (prereg §4)")}
    else:
        out["P8_P1_holds_under_both_readers"] = {
            "held": None, "verdict": "NOT MEASURED -- reader B was not run. "
            "Do not present reader A as if P8 had been measured (prereg §4)."}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    for r in out["readers"]:
        print(f"\n=== {r['reader']}  n={r['n']} ===")
        print(f"{'gold table':12}{'n':>5}{'cell':>8}{'goldtable':>11}{'delta':>8}"
              f"{'win:loss':>10}{'p':>9}")
        for name, _lo, _hi in BUCKETS:
            b = r["by_gold_table_size"][name]
            d = "-" if b["delta"] is None else f"{b['delta']:+.4f}"
            p = "-" if b["p"] is None else f"{b['p']:.4f}"
            wl = "%d:%d" % (b["win"], b["loss"])
            print(f"{name:12}{b['n']:>5}{(b['cell'] or 0):>8.3f}"
                  f"{(b['goldtable'] or 0):>11.3f}{d:>8}{wl:>10}{p:>9}")
        for k, v in r.items():
            if k.startswith("P"):
                print(f"  {k:48} {'HELD' if v.get('held') else 'no':>4}  got={v.get('got')}")
    print("\nP8:", out["P8_P1_holds_under_both_readers"])
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
