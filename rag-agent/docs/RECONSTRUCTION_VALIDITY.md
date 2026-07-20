# What the 99.9% reconstruction number does and does not measure

Status: open question, 2026-07-20. Nothing here changes a result; it records that the
headline reconstruction figure rests on a narrower claim than §3.0 currently implies,
and names the experiment that would settle it.

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

## Scope note

This is not the reconstruction-improvement lever, which is closed (row axis spent, no
"fix → retrieval gain" claim). It is a validity check on a number already in the draft.
