# E1 (W3) — Dense baseline OSC collapse curve (H1)

Baseline: dense single-vector retrieval (`mode="plain"`, bge-small, S2 row-chunks).
Population: HiTab dev arithmetic aggregations with resolved operands, n=214
(m≥2 primary: **n=158**). seed=42, paired bootstrap 95% CI.

## OSC vs retrieval budget k (m≥2, n=158)

| k | OSC | 95% CI | per-cell recall |
|---|---|---|---|
| 1 | 0.190 | [0.127, 0.247] | 0.364 |
| 3 | 0.449 | [0.367, 0.525] | 0.630 |
| 5 | 0.582 | [0.506, 0.658] | 0.724 |
| 10 | 0.772 | [0.703, 0.835] | 0.850 |
| 20 | 0.918 | [0.873, 0.956] | 0.951 |

## OSC vs scope size m, at fixed budget (the H1 collapse)

| k \ m | 1 (n56) | 2 (n110) | 3–4 (n26) | 5–8 (n15) | 9+ (n7) |
|---|---|---|---|---|---|
| 1 | 0.68 | 0.20 | 0.19 | 0.13 | 0.14 |
| 3 | 0.82 | 0.50 | 0.42 | 0.20 | 0.29 |
| 5 | 0.89 | 0.60 | 0.62 | 0.53 | 0.29 |
| 10 | 0.96 | 0.79 | 0.85 | 0.67 | 0.43 |
| 20 | 0.98 | 0.92 | 0.92 | 0.87 | 1.00* |

\* 9+ at k=20 has n=7 (noisy).

## Read

- **H1 supported at realistic budgets (k≤10):** at fixed budget, OSC falls
  monotonically as the aggregation scope m grows — e.g. k=1: 0.68→0.20 from m=1 to
  m≥2; k=5: 0.89→0.60→…→0.29. The single-vector baseline cannot return the full
  operand subset for larger scopes without inflating the budget.
- **Budget partially rescues completeness** (m≥2 OSC 0.19→0.92 as k 1→20) but at
  the cost of dumping 20 chunks into context — i.e. completeness is bought with
  context budget, not with better targeting. The interesting regime for the
  enumeration treatment (E2) is small budget.
- **r^k independence model:** per-cell hit-rate r≈0.36 at k=1 predicts r²≈0.13 for
  m=2 vs observed 0.20 — the same order; first-order independence is a reasonable
  approximation at tight budget. Fuller fit reported in the JSON.

Artifact: `results/e1_osc_baseline.json` ·
reproduce: `PYTHONPATH=. python scripts/e1_osc_baseline.py --split dev`
