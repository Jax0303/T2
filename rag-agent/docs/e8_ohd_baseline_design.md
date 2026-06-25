# E8 — OHD (whole-table structure-aware serialization) as a baseline (design)

OHD (Orthogonal Hierarchical Decomposition, 2602.01969; see `RELATED_DELTA.md`) is
the **nearest representation-side work**: it builds the *same* orthogonal row-tree +
column-tree on HiTab, but uses them to **serialize the *whole* table** (each cell as
`Context → Key → Value`) and feed it all to the LLM — it performs **no retrieval /
cell selection**. That makes it the ideal foil for our central claim:

> **retrieval (minimal complete operand set) reaches OHD-level answer accuracy at a
> fraction of the context, and stays feasible where whole-table serialization does
> not.**

## What we compare

Same fixed LLM solver as E7; arms differ only in what context they build:

| arm | context | selection? |
|---|---|---|
| `ohd_lite` (this design) | **whole table**, each cell `row-path > col-path = value`, row-major **and** col-major orderings | none (OHD-style) |
| `enum_treated` (ours, E7) | retrieved operand subset, same per-cell format | yes |
| `dense_k*`, `oracle` | (as in E7) | — |

`ohd_lite` is a **faithful-enough approximation** of OHD, not their exact system:
we reuse the header trees already in `OriginalTable` (row_path / col_path per cell),
render every numeric cell in OHD's `Context→Key→Value` form, and present both the
row-major and column-major linearizations. We **omit** OHD's learned Orthogonal Tree
Induction and the LLM "semantic arbitrator" that picks the better linearization;
those affect *representation quality*, not the *whole-table-vs-retrieval* axis we
test. (Note honestly in the writeup; if OHD releases code, run it directly.)

## Metrics (the comparison is two-dimensional)

- **Answer accuracy** (numeric match), same scorer/LLM as E7.
- **Context size** — mean cells and mean tokens fed to the LLM. This is the axis
  OHD ignores and we win on.
- **Feasibility / oversize rate** — fraction of queries whose whole-table context
  exceeds the model's token limit (E7 already shows `whole_table` hits the free-tier
  6000-TPM cap → 413). A whole-table method that *cannot run* on a table is a
  first-class result.

## Hypotheses

- **H6 (parity-at-fraction):** `enum_treated` answer accuracy ≈ `ohd_lite` accuracy
  (no significant paired difference) while using ≫ fewer tokens (report the ratio).
- **H6a (scalability):** `ohd_lite` oversize/fail rate grows with table size; our
  retrieved context stays within budget on the same tables → retrieval is the only
  arm that runs on the largest tables.
- **External reference:** OHD report HiTab **60.07 EM** (Qwen2-72b, whole dev) — cite
  as a context line; our numbers use a smaller solver, so this is *reference, not a
  head-to-head win claim*.

## What we will / will NOT claim

- ✅ "At a fixed solver, retrieving the minimal complete operand set matches whole-
  table OHD-style serialization on answer accuracy while using N× fewer tokens, and
  remains feasible on tables where whole-table serialization exceeds the context."
- ✅ Representation and retrieval are **complementary** (OHD's serialization could
  format our retrieved cells).
- ❌ Not "we beat OHD" on their metric/LLM (different solver; `ohd_lite` ≠ full OHD).
- ❌ Not claiming the orthogonal-tree representation as ours (OHD has it).

## Implementation (extend E7)

Add an `ohd_lite` arm to `scripts/e7_retrieval_ablation.py`:
1. cells = all numeric cells (already the `whole_table` cell set).
2. render two serializations from `OriginalTable`:
   - row-major: group by row leaf, lines `row-path > col-path = value`;
   - col-major: group by col leaf, same lineage.
   concatenate both (OHD presents both to the arbitrator); keep the per-cell
   `Context→Key→Value` order.
3. token guard / oversize handling already exists (records `oversize`); report the
   oversize rate explicitly for this arm (it *is* the scalability finding).
4. record accuracy + cells + token estimate; pair vs `enum_treated` (McNemar + CI).

### Status: implemented in `scripts/e7_retrieval_ablation.py`

- `ohd_lite` arm = `ohd_serialize()` (whole table, per-cell `lineage | lineage =
  value`, **row-major + column-major**; omits OHD's learned induction + arbitrator).
- **Dual metrics** recorded for every arm: `accuracy_nm` (numeric-match) **and**
  `accuracy_em` (exact-match, `rag_agent/eval/metrics.exact_match`) — so we line up
  with OHD/HiTab's EM *and* keep the format-robust NM.
- `--baseline` selects the paired arm (E8: `ohd_lite`); ΔNM + bootstrap CI + McNemar
  + ΔEM reported vs it.
- `--max-ctx-tokens` (default 4500) marks too-large contexts `oversize` (skipped, not
  crashed) → the per-arm `n_oversize` **is** the scalability result.

Run (70b solver, fair = same model/metric/population for every arm; separate output
so it doesn't touch the 8b E7 run):
```
PYTHONPATH=. python scripts/e7_retrieval_ablation.py --split dev \
  --llm groq:llama-3.3-70b-versatile \
  --arms enum_treated,ohd_lite,oracle,dense_k10 --baseline ohd_lite \
  --max-ctx-tokens 8000 \
  --out results/e8_ohd_baseline.json --checkpoint results/e8_records.jsonl
```

Caveat (honest): on the free ~6000-TPM tier most `ohd_lite` contexts (~8–9k tokens,
the dual serialization roughly doubles them) are **oversize** → that is the H6a
scalability evidence, but H6 *accuracy* parity can only be read on the subset of
small tables where `ohd_lite` fits. A higher-TPM key removes the restriction. We do
**not** compare to OHD's published 60.07 EM (different model/metric/population) — only
to `ohd_lite` run inside our harness with the identical solver/metric/population.
