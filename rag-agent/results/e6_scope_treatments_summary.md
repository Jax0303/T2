# E6 — structural scope-selection treatments (task 2)

Diagnosis-driven treatments for the row-axis bottleneck, each a row augmentation
over the **same** hybrid-resolved scope (row=embed, col=lexical), measured paired
against the un-augmented hybrid enumeration (`base`). LLM-free. Population: HiTab
dev arithmetic, distinct-cell scope m≥2, **current rebuilt gold (n=161)**.

Reproduce: `PYTHONPATH=. python scripts/e6_scope_treatments.py --split dev --dense`
Artifacts: `results/e6_scope_treatments.json`. Treatments + unit tests:
`rag_agent/retrieve/header_enum.py`, `tests/test_header_enum.py`.

## Why these treatments (not the pre-registered ones)

The pre-registered task-2 levers were parent-subtree enumeration and sibling-group
recognition. The diagnosis (`diag_row_failures_summary.md`) showed those target the
*minority* (≤19/62 row failures); the dominant structure (68%) is **total-row
pairing** — share/ratio queries whose operand set is a sub-scope ∪ a table-level
total row the resolver cannot name. So the primary treatment is total-row
augmentation; sibling expansion is kept as the secondary (pre-registered) lever.

## Results (paired vs hybrid `base`, n=161)

| arm | OSC | ΔOSC vs base | CI95 | McNemar (arm:base) | mean cells | row-cov | col-cov | OSC\|decomp |
|---|---|---|---|---|---|---|---|---|
| base (hybrid enum) | 0.416 | — | — | — | 19.2 | 0.615 | 0.733 | 1.00 |
| **T_total_all** | **0.596** | **+0.180** | [0.124, 0.242] | 29:0 | 30.3 | 0.845 | 0.733 | 1.00 |
| T_total_ratio | 0.584 | +0.168 | [0.118, 0.230] | 27:0 | 29.0 | 0.814 | 0.733 | 1.00 |
| T_subtree | 0.460 | +0.043 | [0.012, 0.075] | 7:0 | 30.9 | 0.665 | 0.733 | 1.00 |
| **T_both** | **0.652** | **+0.236** | [0.174, 0.304] | 38:0 | 39.5 | 0.888 | 0.733 | 1.00 |

## Paired vs the dense single-vector baseline (current gold, same n=161)

Dense baseline OSC: **k=5 = 0.596, k=10 = 0.789** (e6 `--dense` reproduces e2 exactly).

| arm | OSC | ΔOSC vs dense k10 | CI95 | McNemar (arm:dense) |
|---|---|---|---|---|
| base (hybrid enum) | 0.416 | −0.373 | [−0.453, −0.286] | 5:65 |
| T_total_all | 0.596 | −0.193 | [−0.292, −0.087] | 23:54 |
| T_subtree | 0.460 | −0.329 | [−0.410, −0.248] | 6:59 |
| **T_both** | **0.652** | **−0.137** | [−0.236, −0.037] | 24:46 |

- **No treatment beats dense top-k on raw OSC** — every ΔOSC-vs-dense-k10 CI is
  entirely negative. The H2 stance holds: enumeration re-localizes and now *acts on*
  the bottleneck, but does not win raw OSC against similarity ranking at k=10.
- **But the structural levers close ~63% of the gap** (base −0.373 → T_both −0.137)
  via the diagnosed mechanism, and at a smaller cell budget than dense k=10's 10
  row-chunks (~50 cells; cf. E5 dense-top-20 ≈ 99). **T_total_all (0.596) ties dense
  k=5 (0.596) exactly** — at a comparable tight budget the total-augmented
  enumeration equals the dense baseline.

## Reading the numbers (honest)

1. **The diagnosis was right about the lever.** Total-row augmentation alone lifts
   OSC **+0.180** [0.124, 0.242] (row-cov 0.615→0.845), every one of 29 discordant
   queries flips *toward* complete — the single highest-value structural lever.
2. **The pre-registered lever (sibling expansion) is genuinely minor:** **+0.043**
   [0.012, 0.075] — significant but small, and at the *same* cell cost (+12 cells)
   as total augmentation's +0.18. It targets the structure the diagnosis flagged as
   the minority; the data confirm the redirection was warranted, not a hunch.
3. **All ΔOSC are pure-superset gains** (McNemar b:0 — no query ever regresses):
   augmentation only adds cells, so OSC is monotone non-decreasing. The cost is
   **precision**, read as `mean cells`: base 19 → T_both 40 (~2×). This is the
   completeness↔precision tension of E5 re-appearing inside the enumeration arm.
4. **Ratio-gating is nearly a no-op here:** 82.6% of the population reads as a
   share/ratio question, so `T_total_ratio ≈ T_total_all` (−0.012 OSC, −1.3 cells).
   The cue is too broad to be a useful precision filter; unconditional total
   augmentation is simpler and marginally better. A sharper gate (and the
   per-arm precision/OSC frontier) is open work.
5. **The column axis is now the binding constraint.** Every arm leaves col-cov at
   0.733; T_both pushes row-cov to 0.888 but OSC tops out at 0.652 because ~27% of
   queries still miss a *column* leaf. The next bottleneck is column-axis
   decomposition, not row scope.
6. **OSC | decomposition-correct stays exactly 1.00 across all arms** — the
   enumeration invariant (a correctly-bound scope yields the complete operand set)
   is preserved; the treatments raise the *decomposition* hit-rate, not the
   enumeration mechanism.

## Verdict

Total-row augmentation is the correct, diagnosis-targeted fix for the dominant
row-axis failure and gives the largest paired OSC gain at the best cost ratio
(+0.180 vs hybrid base); sibling expansion is a small, confirmed-minor add-on
(+0.043), exactly as the diagnosis predicted. Combined (T_both) they raise row-cov
0.615→0.888 and OSC 0.416→0.652 and **close ~63% of the gap to the dense baseline**
— but do **not** beat dense k=10 on raw OSC (Δ −0.137, CI still negative), and they
~double the cell budget. The enumeration arm now ties dense **k=5** at a comparable
budget. "100% completeness in a *small* set" remains unsolved — these levers buy
completeness with precision (E5 tension), and the residual cap is now the **untouched
column axis** (col-cov 0.733 across all arms), not row scope.
