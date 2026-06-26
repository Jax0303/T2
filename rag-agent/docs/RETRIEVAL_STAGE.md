# Retrieval-stage results (consolidated)

The contribution is evaluated at the **retrieval stage**: given a hierarchical-table
aggregation query, retrieve the operand cells with high **completeness (OSC)** and
**precision (few cells)**. Answer generation is a *separate, model-dependent* stage —
E7 showed an 8b solver floors every retrieval arm at ~0.13 answer accuracy (only the
3-cell oracle reaches 0.61), so retrieval quality is invisible end-to-end until a
stronger solver is used. We therefore report the retrieval objective directly.

Population: HiTab dev arithmetic, distinct-cell scope **m≥2, n=161** (current gold).
All numbers LLM-free, paired on the same queries. Reproduce: `scripts/e2_osc_enum.py`,
`scripts/e6_scope_treatments.py`, `scripts/diag_row_failures.py`,
`scripts/e7_retrieval_ablation.py --dry-run`.

## Consolidated table

| config | OSC | row-cov | col-cov | mean cells |
|---|---|---|---|---|
| dense top-5 (similarity) | 0.596 | — | — | 32 |
| dense top-10 | **0.789** | — | — | 57 |
| dense top-20 | 0.919 | — | — | 98 |
| enum, no treatment | 0.416 | 0.615 | 0.733 | **19** |
| **enum + total-row + sibling** (ours) | **0.652** | **0.888** | 0.733 | 40 |
| enum + cross-encoder column | 0.596 | 0.888 | 0.677 | 31 |
| whole table (recall-first ceiling) | 1.000 | 1.000 | 1.000 | 160 |

## What the retrieval stage shows

1. **Header-tree enumeration is complete-by-construction.** When the query is
   decomposed to the right scope node, enumerating its leaves yields the full operand
   set: **OSC | decomposition-correct = 1.000, flat across scope size m** (E2). The
   H1 collapse of similarity retrieval (dense OSC falls 0.60→0.29 as m grows) is
   *eliminated*. The retrieval problem is **re-localized** from arbitrary-subset
   selection to header-path decomposition.

2. **Row axis — solved by total-row augmentation.** Diagnosis: 68% of row-axis misses
   are share/ratio queries needing an unnamed table **total** row. Adding it (+
   sibling expansion) lifts **row-cov 0.615 → 0.888** and OSC 0.416 → 0.652 (paired
   ΔOSC +0.236 vs the untreated enumeration, CI [0.174, 0.304]). Clear win.

3. **Column axis — cross-encoder is the best selector *at a budget*.** OSC rewards the
   trivial whole-axis dump (col-complete but exploded), so it hides selection quality.
   Measured directly as **col-recall@k** (gold column(s) within the top-k selected),
   the column axis IS a schema-linking problem and a cross-encoder wins at every
   budget (`scripts/col_select_bench.py`, n=161):

   | column selector | col-recall@1 | @2 | @3 |
   |---|---|---|---|
   | lexical (was our default) | 0.267 | 0.398 | 0.472 |
   | bi-encoder embed | 0.267 | 0.565 | 0.677 |
   | **cross-encoder** | **0.398** | **0.609** | **0.727** |

   Diagnosis: when a query names no column, 74% still need exactly **one** column —
   usually a *metric* column ("%", "prevalence per 100,000", "odds ratio") the query
   describes in words; the cross-encoder reads (query, header) jointly and ranks them
   right ("percentage"→"%"). At top-2 it lifts column recall **+0.21 over lexical**.
   It is *not* solved (≈0.73@3; year-ambiguous + multi-column cases remain), but the
   cross-encoder is clearly the right column matcher, and "few columns" is exactly
   what the answer stage needs (so the at-budget metric, not OSC, is the right one).

4. **Completeness vs precision is a frontier, not a point.** Precise (enum, 19 cells)
   → OSC 0.42; complete (whole table, 160 cells) → 1.00; our treated enum sits at
   0.65 / 40 cells. 100% completeness is reachable (recall-first union, E5) but costs
   ~76% of the table. "100% in a *small* set" is the open problem.

## Honest position vs the dense baseline

Enumeration does **not** beat dense top-10 on raw OSC (0.652 < 0.789) — dense buys
completeness with budget (57 cells) and degrades gracefully. Our contribution is
**not** "higher OSC at equal budget"; it is:
- **scope-size robustness** (OSC|decomp flat vs dense's H1 collapse),
- **completeness-by-construction + a 100% guarantee** similarity ranking cannot give,
- a **diagnosis-driven row-axis fix** (total-row, row-cov +0.27),
- and **naming the column axis** as the precise-completeness bottleneck, with a
  cross-encoder that improves precision (cells, explosions) though not yet coverage.

## Limits / open

- **Column completeness** (col-cov 0.733) — exact column selection on unnamed/metric
  axes; cross-encoder helps precision, not coverage. Top open problem.
- **End-to-end answer accuracy** needs a stronger solver than 8b to reveal retrieval
  gains (E7: 8b floors all retrieval arms; oracle 0.61).
- Single dataset (HiTab); selection/comparison queries excluded (gold limitation).
