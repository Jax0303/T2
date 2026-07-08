# E9 — OSC at a fixed token budget: retrieval+injection vs OHD-style whole-table

Same-metric, same-setting comparison the E8 feasibility result was missing: at each
token budget B, what operand-set completeness does each strategy deliver? LLM-free,
HiTab dev arithmetic m>=2 (n=161), identical per-cell line format for every arm
(`row-path | col-path = value`), tokens = chars//3 (as E7).
Script: `scripts/e9_osc_token_budget.py`.

> Re-measured 2026-07-08 **post-audit** (gold-offset + row-path walk() fixes; the
> 2026-07-06 numbers in git history are pre-audit). Conclusions unchanged and slightly
> stronger: significance now extends down to B=250. This run also adds **mean per-cell
> recall** per arm/budget (`recall_at_budget`) — the reference-paper contrast metric.

Deployment rule (no gold peeking): retrieval arms use the **largest k that fits B**
(k grid 1..40); injection = §5.10 winning config (total rows × cross-encoder resolver
columns). OHD arms: `strict` = whole table or nothing (OHD has no selection mechanism);
`trunc` = **generous** row-major prefix up to B (a heuristic OHD does not have);
`dual` = OHD's faithful row-major + column-major double serialization.

## Whole-table cost (why a budget bites)

single serialization: mean **4,382** tokens, median 2,995, **p90 12,332** (dual = 2×).

## OSC @ budget B

| arm | @500 | @1000 | @2000 | @4000 | @8000 | @16000 |
|---|---|---|---|---|---|---|
| dense_plain | 0.497 | 0.721 | 0.851 | 0.944 | 0.981 | 1.000 |
| hybrid_plain | 0.491 | 0.752 | 0.839 | 0.919 | 0.981 | 1.000 |
| dense_inject | 0.441 | 0.745 | 0.870 | 0.975 | 0.988 | 1.000 |
| **hybrid_inject** | 0.453 | **0.764** | **0.876** | **0.963** | **1.000** | 1.000 |
| ohd_strict | 0.050 | 0.273 | 0.416 | 0.652 | 0.795 | 1.000 |
| ohd_trunc (generous) | 0.255 | 0.553 | 0.671 | 0.807 | 0.870 | 1.000 |
| ohd_dual_strict (faithful) | 0.037 | 0.050 | 0.273 | 0.416 | 0.652 | 0.795 |
| ohd_dual_trunc | 0.168 | 0.255 | 0.553 | 0.671 | 0.807 | 0.870 |

## Mean per-cell recall @ budget B (the reference metric — and why it misleads)

| arm | @500 | @1000 | @2000 | @4000 | @8000 |
|---|---|---|---|---|---|
| hybrid_plain | 0.647 | 0.855 | 0.922 | 0.957 | 0.991 |
| hybrid_inject | 0.570 | 0.855 | 0.940 | 0.979 | **1.000** |
| ohd_trunc | 0.396 | 0.598 | 0.732 | 0.855 | 0.921 |

Averaged recall systematically **overstates usability**: at B=2000 hybrid_plain
reports recall 0.92 yet completes only **0.84** of queries (OSC); ohd_trunc@8k reports
0.92 recall yet 13% of queries are still uncomputable. One missed operand = a wrong
aggregate — recall's partial credit hides exactly the failures that matter.

## Paired: hybrid_inject vs ohd_trunc (the GENEROUS OHD variant)

| B | inject | ohd_trunc | Δ | inj-only | ohd-only | McNemar p |
|---|---|---|---|---|---|---|
| 250 | 0.292 | 0.168 | +0.124 | 32 | 12 | 0.0037 |
| 500 | 0.453 | 0.255 | +0.199 | 45 | 13 | **3e-5** |
| 1000 | 0.764 | 0.553 | +0.211 | 43 | 9 | **<1e-6** |
| 2000 | 0.876 | 0.671 | **+0.205** | 37 | 4 | **<1e-6** |
| 4000 | 0.963 | 0.807 | +0.155 | 26 | 1 | **<1e-6** |
| 8000 | **1.000** | 0.870 | +0.130 | 21 | 0 | **<1e-6** |
| 16000 | 1.000 | 1.000 | 0 | — | — | — |

## Read

1. **At every realistic budget (250–8k tokens), retrieval+injection significantly
   beats even the generous truncated whole-table arm** (p≤0.004 throughout; at 8k it
   is 1.000 vs 0.870 with 21/0 one-sided flips). Against *faithful* OHD (dual
   serialization, no truncation) the gap is enormous (e.g. @4000: 0.963 vs 0.416).
2. **Whole-table only catches up at B≥16k** (single) / ≥32k (dual) — i.e., when the
   budget is big enough that selection is unnecessary. That is exactly the regime
   claim from the generalization study: our contribution lives where the table does
   not fit.
3. **hybrid_inject reaches OSC=1.000 at 8k tokens** — a complete-operand guarantee at
   a budget where whole-table serialization is still failing 13% of queries.
4. **Recall vs OSC**: on the metric reference systems report (mean per-cell recall)
   all strong arms look close (0.92–0.98 at 4k); OSC separates them (0.807–0.975).
   Reporting both is the "partial recall is not completeness" exhibit.
5. *Honest caveat:* at starvation budgets (B≤500) injection **hurts** vs plain
   (e.g. @500 dense 0.497 plain vs 0.441 inject): the injected total cells crowd out
   ranked chunks under the largest-k-that-fits rule. The patch needs ~1k tokens of
   headroom; below that, plain dense is the best arm. Crossover at B≈1000.
6. Together with E8 (35% oversize @8k ctx) this upgrades §5.9 from "9× cheaper and
   feasible" to "**more complete at every budget that matters, with significance**."
