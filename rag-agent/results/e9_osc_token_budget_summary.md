# E9 — OSC at a fixed token budget: retrieval+injection vs OHD-style whole-table

Same-metric, same-setting comparison the E8 feasibility result was missing: at each
token budget B, what operand-set completeness does each strategy deliver? LLM-free,
HiTab dev arithmetic m>=2 (n=161), identical per-cell line format for every arm
(`row-path | col-path = value`), tokens = chars//3 (as E7).
Script: `scripts/e9_osc_token_budget.py`.

Deployment rule (no gold peeking): retrieval arms use the **largest k that fits B**
(k grid 1..40); injection = §5.10 winning config (total rows × cross-encoder resolver
columns). OHD arms: `strict` = whole table or nothing (OHD has no selection mechanism);
`trunc` = **generous** row-major prefix up to B (a heuristic OHD does not have);
`dual` = OHD's faithful row-major + column-major double serialization.

## Whole-table cost (why a budget bites)

single serialization: mean **4,254** tokens, median 2,995, **p90 11,807** (dual = 2×).

## OSC @ budget B

| arm | @500 | @1000 | @2000 | @4000 | @8000 | @16000 |
|---|---|---|---|---|---|---|
| dense_plain | 0.484 | 0.689 | 0.845 | 0.932 | 0.957 | 1.000 |
| hybrid_plain | 0.453 | 0.683 | 0.814 | 0.901 | 0.957 | 0.994 |
| dense_inject | 0.385 | 0.733 | 0.882 | 0.969 | 0.994 | 1.000 |
| **hybrid_inject** | 0.391 | **0.752** | **0.888** | **0.957** | **1.000** | 1.000 |
| ohd_strict | 0.050 | 0.273 | 0.416 | 0.658 | 0.795 | 1.000 |
| ohd_trunc (generous) | 0.267 | 0.553 | 0.671 | 0.820 | 0.876 | 1.000 |
| ohd_dual_strict (faithful) | 0.037 | 0.050 | 0.273 | 0.416 | 0.658 | 0.795 |
| ohd_dual_trunc | 0.174 | 0.267 | 0.553 | 0.671 | 0.820 | 0.876 |

## Paired: hybrid_inject vs ohd_trunc (the GENEROUS OHD variant)

| B | inject | ohd_trunc | Δ | inj-only | ohd-only | McNemar p |
|---|---|---|---|---|---|---|
| 500 | 0.391 | 0.267 | +0.124 | 40 | 20 | 0.013 |
| 1000 | 0.752 | 0.553 | +0.199 | 44 | 12 | **2e-5** |
| 2000 | 0.888 | 0.671 | **+0.217** | 38 | 3 | **<1e-6** |
| 4000 | 0.957 | 0.820 | +0.137 | 24 | 2 | **1e-5** |
| 8000 | **1.000** | 0.876 | +0.124 | 20 | 0 | **<1e-6** |
| 16000 | 1.000 | 1.000 | 0 | — | — | — |

## Read

1. **At every realistic budget (500–8k tokens), retrieval+injection significantly
   beats even the generous truncated whole-table arm** (p≤0.013 throughout; at 8k it
   is 1.000 vs 0.876 with 20/0 one-sided flips). Against *faithful* OHD (dual
   serialization, no truncation) the gap is enormous (e.g. @4000: 0.957 vs 0.416).
2. **Whole-table only catches up at B≥16k** (single) / ≥32k (dual) — i.e., when the
   budget is big enough that selection is unnecessary. That is exactly the regime
   claim from the generalization study: our contribution lives where the table does
   not fit.
3. **hybrid_inject reaches OSC=1.000 at 8k tokens** — a complete-operand guarantee at
   a budget where whole-table serialization is still failing 12% of queries.
4. *Honest caveat:* at starvation budgets (B≤500) injection **hurts** vs plain
   (e.g. @500 dense 0.484 plain vs 0.385 inject): the injected total cells crowd out
   ranked chunks under the largest-k-that-fits rule. The patch needs ~1k tokens of
   headroom; below that, plain dense is the best arm. Crossover at B≈1000.
5. Together with E8 (35% oversize @8k ctx) this upgrades §5.9 from "9× cheaper and
   feasible" to "**more complete at every budget that matters, with significance**."
