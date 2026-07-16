# Plan — Header-tree reconstruction accuracy (diagnostic-gated)

Status: **planned, not started.** Recorded 2026-07-16. Do the two gate diagnostics
first; only implement the improvements if the gates pass.

## Why

Cell localization has two stages: (1) reconstruct the table's header hierarchy, then
(2) address each cell by its (row_path, col_path). Stage-1 accuracy, measured against
HiTab gold trees, is **column 96.7% / row 87.6% exact-match** (MultiHiertt has no gold
tree — proxy ≈ 90%). Reconstruction is deterministic (100 identical runs), so the loss
is not variance but **per-table error**: ~3% of columns, ~12% of rows get a wrong path.
These errors degrade **only S2/S3 (our method)**, never the flat baseline, so closing
them is directly aligned with the "show a performance gain" goal.

Error is not uniform across the pipeline:
- **Merged-cell expansion (colspan/rowspan): essentially exact** — driven by explicit
  HTML attributes, no guessing. Not a target.
- **Header-boundary detection (`guess_n_header_rows`, "first row ≥50% numeric"): the
  main error source** — a single fragile heuristic.
- **Row hierarchy (section headers + indentation): the weak axis (87.6%)** — signals are
  ambiguous, so boundary/nesting misfires more often than on columns.

## Gate diagnostics (do these FIRST — they decide whether the work is worth it)

**G1 — Is header markup available?** Check whether the raw MultiHiertt/HiTab HTML carries
explicit header cells (`<th>`) or other structural markup. If yes, boundary detection can
become attribute-driven (near-exact, like colspan) instead of a numeric guess — a cheap
win that could push column accuracy toward ~100%.

**G2 — Do reconstruction errors overlap the gold operands?** Measure the share of
mis-reconstructed cells that are actually gold operand cells for some query. If the
overlap is small, fixing reconstruction will **not** move set-EM/answer metrics, and this
whole effort should be dropped in favor of another lever (e.g. reranking). This is the
go/no-go test.

## Improvements (only if gates pass), cheap → expensive

**A. Multi-signal header-boundary detection** (cheapest, highest-leverage)
- Use `<th>` markup when present (from G1).
- Column-type homogeneity: a data column is type-consistent top-to-bottom; boundary =
  where homogeneity starts.
- colspan pattern: header rows span, data rows do not.

**B. Row-hierarchy signals** (row axis is the weak 87.6%)
- Indentation detection (`&nbsp;`/leading spaces) for parent–child nesting.
- Section-header-row detection (only the label column filled, rest blank → a parent).
- Estimate `n_header_cols` per table instead of the fixed `=1`.

**C. Self-verification / ensemble** (medium)
- Vote across 2–3 heuristics; if reconstruction assigns the **same address to two distinct
  cells**, flag and re-parse.

**D. Learned / LLM-based structure parsing** — **OUT OF SCOPE before the July deadline.**
A new research thread, not a component fix.

## Constraints / honesty

- Respect the dropped merged-cell-**complexity** research direction (professor's call):
  this plan is a **quality improvement of the existing reconstructor**, not a complexity
  study, and stays limited to A/B low-risk changes.
- Keep the HiTab (exact-match, upper bound) vs MultiHiertt (proxy) measurement distinction
  — do not compare the two accuracy numbers directly.

## Decision rule

Run G1 + G2 (≈ half a day). Proceed to **A** only if G2 shows meaningful gold-operand
overlap **and** (ideally) G1 finds usable markup. Otherwise drop reconstruction and move
the effort to a different lever.
