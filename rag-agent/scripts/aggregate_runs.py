#!/usr/bin/env python3
"""Aggregate multiple result JSONs into one comparison table.

Usage:
  python aggregate_runs.py <name1>=<path1>.json <name2>=<path2>.json ...
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def main():
    if len(sys.argv) < 2:
        print("usage: aggregate_runs.py NAME=path.json [NAME=path.json ...]")
        sys.exit(1)

    runs = []
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        runs.append((name, json.load(open(path))))

    classes = ["multi_op_formula", "arithmetic_agg", "pair_or_topk_arg",
               "single_arg", "comparison_or_count"]

    header = ["class"] + [f"{n}_{m}" for n in (r[0] for r in runs) for m in ("R@1", "NM", "sym_c")]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---:"] * (len(header))) + "|")

    for cls in classes + ["__overall__"]:
        row = [cls if cls != "__overall__" else "OVERALL"]
        for name, data in runs:
            src = data["overall"] if cls == "__overall__" else data["per_class"].get(cls, {})
            n = src.get("n", 1) or 1
            row += [f"{src.get('R@1_final',0)/n:.3f}",
                    f"{src.get('NM',0)/n:.3f}",
                    f"{src.get('symbolic_correct',0)/n:.3f}"]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
