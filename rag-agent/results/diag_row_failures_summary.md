# Row-axis failure diagnosis (task 1) — what structure breaks enumeration?

Tears apart **every** row-axis failure of the hybrid resolver (E2 enumeration,
`row_cov == 0`: not all gold row leaves enumerated) on the primary population
(HiTab dev arithmetic, distinct-cell scope m≥2). Goal: before guessing a treatment,
measure *which structure* dominates the residual row-axis bottleneck — the part
the embedding idea did **not** close (it closed lexical/depth, §RESULTS).

Reproduce: `PYTHONPATH=. python scripts/diag_row_failures.py --split dev`
Artifacts: `results/diag_row_failures.json`, `results/diag_row_failures.jsonl`.

> Note on population: recomputed on the **current** rebuilt gold
> (`results/operand_gold.jsonl`), giving n=161 and hybrid row-coverage 0.615.
> The committed `e2_osc_enum_hybrid.json` (n=158, row-cov 0.582) predates the gold
> rebuild; the taxonomy is robust to this ±3-query drift.

## Headline

**62 / 161 queries fail the row axis** (row-cov 0.615). The failures are **not**
dominated by "which/how-many siblings" as expected. They are dominated by a
different structure entirely:

| refined bucket | n | % of row-fail | what it is |
|---|---|---|---|
| **total_pairing** | **42** | **68%** | a missed operand row is a table/section **total** (share-of-total / ratio). Operand set = a sub-scope ∪ a total row that sits at a *different* header level (often an **empty / unparsed** left-header path), so a header-text resolver can never bind it. |
| sibling_subset | 9 | 15% | same immediate parent, gold is a *strict subset* of the children → genuine "which siblings" selection. |
| cross_parent | 7 | 11% | genuine multi-entity cross-cut (no total row). |
| parent_expandable | 4 | 6% | gold = *all* numeric children of one parent → blunt subtree expansion fixes it. |

`total_pairing` is overwhelmingly the **`div` (ratio/share)** aggregation:
32 of 42. The canonical case: *"what percentage of total R&D did large companies
account for?"* → operands = {the large-companies leaf rows} ∪ {the **total** row}.
The total row has header path `''` (unparsed) or `… > total` / `total population`,
which no amount of semantic row-embedding can match to the query words.

## Why the pre-registered task-2 hypotheses were aimed at the minority

The plan proposed (a) parent-subtree enumeration and (b) sibling-group recognition.
Oracle **row-axis** recovery ceiling per lever (would the row axis become covered
if the lever fired perfectly; row-axis only, OSC still needs col coverage):

| lever (oracle) | rows recovered / 62 |
|---|---|
| **total-row augmentation** (add every total-like row) | **37** |
| parent/LCP-subtree expansion | 19 |
| either | 54 |
| neither (genuine hard cross-cut) | 8 |

- The two pre-registered levers (subtree / sibling) map to **parent_expandable +
  sibling_subset = 13 queries**, and the broader LCP-expansion oracle tops out at
  **19** — the *minority*.
- The unanticipated **total-row augmentation** alone recovers **37** (single
  largest lever) and is cheap (a table has very few total rows).
- Combined oracle ceiling is **54/62 (87%)** row-axis recovery; only 8 are
  irreducibly hard (genuine cross-cuts with no total and no common parent).

## Other facts (honest caveats)

- **total_miss 37%** (23/62 caught *zero* gold rows) vs partial 63%. Many total_miss
  are `gold = ['', '']` — both operands are header-less total/aggregate rows; these
  are pure structural/value targets with nothing for any header resolver to grab.
- **row_fallback rate 0** — the resolver always matched *something* on the row axis,
  it just matched the wrong/too-narrow scope; failures are mis-selection, not "no
  match → whole axis".
- **LCP-parent expansion is expensive**: when it recovers, it adds a mean of **+8.8**
  extra numeric rows — a precision collapse, consistent with E5's completeness↔
  precision tension. Total-row augmentation adds ~1 row instead.
- **Column axis is a separate, non-trivial contributor**: 32 queries have the row
  axis covered but the **column** axis missed (`col_only_failures`). Not in scope
  here, but it caps any row-only treatment's OSC gain.

## Redirected task-2 hypotheses (diagnosis-driven)

1. **T_total — ratio-aware total-row augmentation (primary).** When enumerating a
   scope, also include the numeric cells of table/section **total** rows (within the
   matched columns). Targets 68% of failures; ~1-row precision cost. Hypothesis:
   largest paired ΔOSC of any single lever, at near-zero precision cost.
2. **T_subtree — parent/sibling-group expansion (secondary).** Detect an aggregation
   over a parent node and enumerate its full child group. Targets parent_expandable
   (clean) + sibling_subset (at precision cost). Hypothesis: smaller ΔOSC, larger
   cell-count cost than T_total.
3. Honest report: OSC, OSC | decomposition-correct, mean cells (precision), paired
   bootstrap 95% CI vs the hybrid enumeration baseline. No raw-win inflation.

The residual after both (~8 genuine cross-cuts + the 32 column-axis misses) is the
true "structural scope selection" hard core.
