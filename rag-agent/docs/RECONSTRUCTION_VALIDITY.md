# What the 99.9% reconstruction number does and does not measure

Status: **settled, 2026-07-21** — the experiment proposed below was run
(`scripts/tree_reconstruct_hitab_raw.py`). Result summary up front; the original
analysis follows unchanged.

| on the real grid | column axis | row axis |
|---|---|---|
| round trip (`tree_reconstruct_hitab.py`) | .9991 | .9996 |
| **real grid, boundary known** (dev, 424 tables) | **.9749** | **.5781** |
| **real grid, boundary known** (train, 2,043 tables) | **.9757** | **.5816** |
| real grid, boundary guessed (dev) | .8819 | .5614 |

Boundary detection, which scores 1.000 on the synthetic grid, scores **.7146** on real
grids.

The row figure is not an algorithm failure — it splits cleanly in two, and the split is
the whole finding (train numbers; dev matches):

| row hierarchy fits the stub-column block? | tables | paths | row exact match |
|---|---|---|---|
| yes | 1,216 | 18,933 | **.9946** |
| no  | 827 | 16,130 | **.0968** |

Where the grid can express the row depth, the reconstructor hits the round-trip number.
Where it cannot, it fails almost totally — and it cannot in **41%** of tables, because
HiTab encodes deep row hierarchies as *parent rows inside a single stub column*
(`institution / all institutions / top 20 institutions / johns hopkins u.` all sit in
column 0, flush, with no indentation surviving in `texts`), while `flatten_to_grid`
renders an *n*-deep row hierarchy as *n* separate stub columns. The round trip scored a
grid shape that 41% of real HiTab tables do not have, which is exactly why it could not
see this.

Verdict against the two outcomes named at the bottom of this file: the column axis lands
close to the expectation (.975 vs .999, a real but modest drop concentrated at depth 3,
.878); the row axis takes the "materially lower" branch, and **.9996 cannot stand in §3.0
as a row-axis accuracy**.

Consequences already applied: §3.0 restated, `docs/sentence_accuracy_table.html` caveated.
Consequence *not* claimed: this does not reopen the reconstruction-improvement lever. The
41% deficit is missing information in the input, not a fixable decoder — recovering it
needs a signal `texts` does not carry (HTML indentation/class markup, or `merged_regions`).

---

## The claim as it currently reads

`docs/PAPER_DRAFT.md` §3.0 reports header-tree reconstruction at
**col 0.9991 / row 0.9996 exact match** on HiTab dev (540 tables, 4,413 column paths /
8,944 row paths), from `results/tree_reconstruct_hitab.json`. It is read as: *the
reconstructor recovers hierarchical header trees from 2D grids at ~99.9% accuracy.*

## What the experiment actually does

`scripts/tree_reconstruct_hitab.py` says so in its own docstring — it *"does the opposite
of what the real pipeline does"* — and the result file names the population
`hitab_synthetic_flatten`. The measurement is a round trip:

1. **Encode.** `flatten_to_grid()` takes the *already-known gold* header paths and renders
   a synthetic "blank-after-first" grid: a label at depth `d` is written only when the full
   prefix `path[:d+1]` differs from the previous column's/row's
   (`tree_reconstruct_hitab.py:47`, `:56`).
2. **Decode.** `_hierarchical_carry()` forward-fills each depth's last non-blank label and
   resets all deeper carries whenever a shallower one changes
   (`rag_agent/reconstruct/header_grid.py:144`).

Step 2 is the exact inverse of step 1. The test encodes with `R⁻¹` and decodes with `R`,
so a high score is close to definitional. The residual 0.1% is where the round trip is not
perfectly invertible — the over-extended-path cases on value-less rows documented in
d2ce3a8, e.g. gold `['immigrant category']` rebuilt as
`['immigrant category', 'refugee']`.

Two further conditions are handed over rather than solved:

- **The header/data boundary is given from gold** (`boundary_mode: "known (ground truth)"`).
- The `--guess-boundary` variant (`results/tree_reconstruct_hitab_guessed.json`) scores
  boundary detection at **1.0** with bit-identical path numbers — but it is guessing on the
  same synthetic grid, which `guess_n_header_rows` is well matched to. It is not evidence
  about real grids either.

**Supported claim:** HiTab's gold trees are expressible in blank-after-first form, and the
decoder inverts that encoding. The algorithm is self-consistent.

**Unsupported claim:** the reconstructor reads real, scraped 2D grids at 99.9%.

## Why this is the MultiHiertt question too

The recurring expectation is that MultiHiertt "should also come out near 99%." It cannot,
for two independent reasons:

1. MultiHiertt ships no gold header tree, so exact match is not merely hard but undefined.
   The `segment_coverage` proxy in `scripts/tree_reconstruct_multihiertt.py` is precision-only
   and hardcodes `n_header_cols=1` (99.9% of scored row paths are depth 1) — already
   documented as unquotable in §3.0.
2. Even with gold, 99.9% would be the wrong expectation, because MultiHiertt grids were
   never produced by `flatten_to_grid`. The round trip transfers nothing to inputs it did
   not generate.

## The experiment that would settle it

HiTab ships the real source grids alongside the gold trees, and the pipeline does not use
them. `bench/hitab.py` reads the parsed `hmt` trees only:

```
/mnt/d/hart_data/hitab/HiTab/data/tables/tables/raw/*.json   (3,597 files)
  keys: texts, merged_regions, top_header_rows_num,
        left_header_columns_num, top_root, left_root
```

`texts` is the genuine 2D grid; `top_root`/`left_root` are the gold trees. So the
non-circular measurement is available in full:

> **real `texts` grid → `reconstruct_col_paths` / `reconstruct_row_paths` → exact match against gold `top_root`/`left_root`**, with the boundary guessed by `guess_n_header_rows`, not read from `top_header_rows_num` / `left_header_columns_num`.

Outcomes:

- **~99%** → the expectation was right, §3.0 gets a genuine accuracy number instead of a
  round-trip one, and extrapolating to MultiHiertt becomes defensible.
- **Materially lower** → the current 0.9991/0.9996 cannot stand in §3.0 as written and must
  be relabelled as an algorithm self-consistency check.

The presence of a separate `merged_regions` field is a hint that raw grids encode spans
differently from blank-after-first. Real tables also carry section-label rows, subtotal
rows, and footnote rows interleaved into the body — none of which `flatten_to_grid` ever
produces.

## How it was actually run (deviations from the sketch above)

Three things had to change, each of which would otherwise have charged the reconstructor
for someone else's representation choice:

- **Gold is the `hmt` parse, not the raw file's `top_root`/`left_root`.** The raw trees are
  trees over header *cells*, so a header merged across two data columns appears as one node
  on the first column only and leaves the second column's path a segment short. On table
  1017 `merged_regions` merges "percent" across columns 2–3: the raw tree gives column 3
  `(hirings, recruit)` where hmt — the published parse the whole pipeline treats as ground
  truth — gives `(hirings, recruit, percent)`. Scoring against the raw tree would penalise
  span resolution that the real gold says is correct.
- **The header block comes from the first data line, not the header tree's extent.** Some
  tables put a row-tree ancestor inside the data-column region (table 1045 has "ratio" at
  column 1, which is also a data column), so `max(left_cols)+1` pushes `n_header_cols` past
  the first data column. `min(data cols)` / `min(data rows)` is the robust definition.
- **Grid↔gold line correspondence is verified, not assumed.** The hmt data matrix is
  data-only and drops lines the grid keeps (an ancestor occupying a row of its own), so gold
  lines are matched onto grid lines monotonically by leaf label — visible in the grid; only
  the *path* is what the reconstructor must infer — then checked by value equality against
  the hmt data matrix at ≥90%. Unverifiable tables are **excluded, not guessed**: dev scores
  424/540 (116 excluded), train 2,043. The excluded tables are a coverage caveat on the
  numbers above.

`top_header_rows_num` is unused: over all 3,597 tables it exceeds the tree-derived
header-row count by exactly 1 in 3,596 of them.

## Scope note

This is not the reconstruction-improvement lever, which is closed (row axis spent, no
"fix → retrieval gain" claim). It is a validity check on a number already in the draft.
