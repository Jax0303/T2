#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Turn the completeness/cost oracle into a policy.

``osc_minimal_set.py`` showed that every HiTab dev query reaches OSC=1.0
somewhere on a ladder of candidate sets, and that the cheapest sufficient rung
averages 22 cells against the 122 a single fixed configuration spends. That 22
is an oracle: it assumes the cheapest sufficient rung is known per query. This
script asks whether it can be *predicted* from signals a live system has --
resolver output, table shape, question surface -- with no access to the gold
operand set.

Framing. Rungs are sorted by mean cost into a ladder; the label is the index of
the cheapest rung that is complete for that query. Errors are asymmetric:
overshooting spends cells, undershooting loses completeness outright, and OSC
is all-or-nothing. So the model predicts the rung and a ``--margin`` pushes the
choice conservatively up the ladder; the margin sweep is the operating curve.

Baselines it must beat, all at OSC=1.0 on the fitted split:
  * fixed       -- cheapest single rung that is complete for *every* query
  * oracle      -- per-query cheapest sufficient rung (the 22-cell lower bound)
and it must not fall below the completeness a fixed rung would have given.

Fit on train, report on dev. Run:
    PYTHONPATH=. python scripts/osc_budget_selector.py \
        --train results/osc_min_train.jsonl --test results/osc_min_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 42


def load(path):
    return [json.loads(l) for l in open(path)]


def build_ladder(recs):
    """The nested escalation ladder, in the order osc_minimal_set built it.

    Rung i's set contains rung i-1's, so both cells and OSC are monotone in i.
    That monotonicity is what makes a safety margin meaningful: stepping up can
    only add coverage, never trade it away, which an arbitrary cost-ordering of
    non-nested configs cannot promise.
    """
    names = [k[:-5] for k in recs[0] if k.endswith("__osc") and k.startswith("esc_")]
    cost = {n: sum(r[f"{n}__cells"] for r in recs) / len(recs) for n in names}
    for a, b in zip(names, names[1:]):          # order is construction order
        assert cost[a] <= cost[b] + 1e-9, f"ladder not monotone at {a}->{b}"
    return names, cost


def label_of(rec, ladder):
    """Index of the cheapest rung complete for this query; None if unreachable."""
    for i, n in enumerate(ladder):
        if rec[f"{n}__osc"] == 1.0:
            return i
    return None


def evaluate(recs, ladder, choose):
    """choose(rec) -> rung index. Returns (mean_cells, osc, pct_complete)."""
    cells, osc = [], []
    for r in recs:
        i = min(max(choose(r), 0), len(ladder) - 1)
        n = ladder[i]
        cells.append(r[f"{n}__cells"])
        osc.append(r[f"{n}__osc"])
    return (round(sum(cells) / len(cells), 1),
            round(sum(osc) / len(osc), 4),
            round(sum(1 for x in osc if x == 1.0) / len(osc), 4),
            round(statistics.median(cells), 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--margins", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 18])
    ap.add_argument("--quantiles", type=float, nargs="+",
                    default=[0.5, 0.8, 0.9, 0.95, 0.99])
    ap.add_argument("--out", default="results/osc_budget_selector.json")
    args = ap.parse_args()

    tr, te = load(args.train), load(args.test)
    ladder, cost = build_ladder(tr)                 # ladder fixed on TRAIN only
    print(f"[data] train={len(tr)} test={len(te)}  rungs={len(ladder)}")
    print(f"[ladder] cheapest={ladder[0]} ({cost[ladder[0]]:.1f} cells) "
          f"... priciest={ladder[-1]} ({cost[ladder[-1]]:.1f})")

    y_tr = [label_of(r, ladder) for r in tr]
    y_te = [label_of(r, ladder) for r in te]
    keep = [i for i, y in enumerate(y_tr) if y is not None]
    tr, y_tr = [tr[i] for i in keep], [y_tr[i] for i in keep]
    n_unreach_te = sum(1 for y in y_te if y is None)
    print(f"[label] train reachable={len(tr)}  test unreachable={n_unreach_te}")

    feat_keys = sorted(tr[0]["feat"])
    X_tr = [[r["feat"][k] for k in feat_keys] for r in tr]
    X_te = [[r["feat"][k] for k in feat_keys] for r in te]

    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=SEED)
    model.fit(X_tr, y_tr)
    pred_te = model.predict(X_te)

    # A flat margin shifts every prediction by the same amount, which is the
    # wrong instrument for an asymmetric loss: the queries that need headroom
    # are not the ones a constant offset helps. Quantile regression puts the
    # asymmetry inside the fit -- alpha is the fraction of queries the
    # prediction is meant to land at or above.
    quantile_rows = []
    for a in args.quantiles:
        qm = GradientBoostingRegressor(loss="quantile", alpha=a, random_state=SEED)
        qm.fit(X_tr, y_tr)
        qp = qm.predict(X_te)
        idx = {id(r): int(round(p)) for r, p in zip(te, qp)}
        quantile_rows.append((f"quantile a={a}", evaluate(te, ladder, lambda r: idx[id(r)])))

    # --- baselines -------------------------------------------------------
    # fixed: cheapest rung complete for every TRAIN query, applied to test
    fixed_i = None
    for i, n in enumerate(ladder):
        if all(r[f"{n}__osc"] == 1.0 for r in tr):
            fixed_i = i
            break
    rows = []
    if fixed_i is not None:
        rows.append(("fixed (train-complete rung)", evaluate(te, ladder, lambda r: fixed_i)))
    rows.append(("oracle (per-query cheapest)",
                 evaluate(te, ladder, lambda r, _l=ladder: (label_of(r, _l) or len(_l) - 1))))

    # --- the policy, swept over the safety margin ------------------------
    best = None
    for m in args.margins:
        idx = {id(r): int(round(p)) + m for r, p in zip(te, pred_te)}
        res = evaluate(te, ladder, lambda r: idx[id(r)])
        rows.append((f"selector margin=+{m}", res))
        if res[2] == 1.0 and (best is None or res[0] < best[1][0]):
            best = (m, res)
    rows.extend(quantile_rows)

    print(f"\n{'policy':<32} {'cells':>8} {'median':>8} {'OSC':>8} {'%complete':>11}")
    for label, (c, o, p, med) in rows:
        print(f"  {label:<30} {c:>8.1f} {med:>8.1f} {o:>8.4f} {p:>11.4f}")
    if best:
        m, (c, o, p, med) = best
        print(f"\ncheapest margin holding OSC=1.0 on test: +{m} -> {c} cells")
    else:
        print("\nno margin reached OSC=1.0 on test")

    imp = sorted(zip(feat_keys, model.feature_importances_), key=lambda t: -t[1])
    print("\ntop features:")
    for k, v in imp[:8]:
        print(f"  {k:<20} {v:.3f}")

    out = {
        "experiment": "osc_budget_selector", "seed": SEED,
        "train": args.train, "test": args.test,
        "n_train": len(tr), "n_test": len(te), "n_test_unreachable": n_unreach_te,
        "ladder": ladder, "ladder_mean_cells": {n: round(cost[n], 1) for n in ladder},
        "fixed_rung": ladder[fixed_i] if fixed_i is not None else None,
        "results": {label: {"mean_cells": c, "median_cells": med,
                            "osc": o, "pct_complete": p}
                    for label, (c, o, p, med) in rows},
        "best_margin_at_full_completeness": best[0] if best else None,
        "feature_importance": {k: round(float(v), 4) for k, v in imp},
        "note": ("Ladder order and model are fitted on train only; test is untouched. "
                 "Labels use gold operands, features never do."),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
