# E3 (W5) — Causal isolation of header depth d (H2 causality)

Holding data, leaf vocabulary, and scope size fixed, isolate the effect of header
*depth* on enumeration completeness. Population: HiTab dev arithmetic m≥2, n=158.
LLM-free. Two tests:

- **(A) Observational** — enumeration OSC / axis coverage stratified by depth d.
- **(B) Synthetic depth control** — leaf-flatten every table to depth 1 (keep only
  the leaf token of each row/col path, drop ancestor levels; same data, same leaf
  vocab, no tree), re-resolve + re-enumerate, paired vs original.

## (A) Observational — deeper tables enumerate worse

| depth | n | OSC enum | row-cov | col-cov | mean cells |
|---|---|---|---|---|---|
| d=2 | 46 | 0.391 | 0.630 | 0.696 | 17.9 |
| d≥3 | 111 | 0.306 | 0.504 | 0.739 | 17.1 |

Deeper header trees show lower enumeration OSC and lower row-axis coverage —
consistent with depth driving the decomposition bottleneck, but observational
(depth is confounded with domain/scope here).

## (B) Synthetic leaf-flatten (d→1) — the causal test (paired, n=158)

| | original (d=2/3) | flattened (d=1) | Δ |
|---|---|---|---|
| OSC enum | 0.335 | **0.570** | **+0.234** |
| col-axis coverage | 0.728 | 0.930 | +0.203 |
| row-axis coverage | 0.544 | 0.601 | +0.057 |
| mean enum cells | 17.2 | 37.9 | ×2.2 |

By original depth:

| orig depth | n | OSC orig → flat | row-cov orig → flat |
|---|---|---|---|
| d=2 | 46 | 0.391 → **0.674** | 0.630 → 0.674 |
| d≥3 | 111 | 0.306 → **0.522** | 0.504 → 0.568 |

**Read — depth causally suppresses enumeration completeness.** Collapsing the
header tree to a single level (same data, same leaf words) raises enumeration OSC
by +0.23, driven mainly by the column axis (coverage 0.73→0.93). The effect is
present at both d=2 and d≥3. Because only the tree structure is removed, this
isolates header *depth* — not domain or vocabulary — as a cause of the
operand-set-completeness difficulty.

### Mechanism and honest caveat
Multi-level paths force the resolver to match a query against deep, ancestor-laden
header strings; the leaf token that the query actually names is buried, so the
fuzzy ranker mis-selects paths and the enumerated scope misses operands. Flattening
exposes the leaf directly. But flattening also **broadens** each predicate (a
single leaf token matches more cells): the flattened scope is 2.2× larger (37.9 vs
17.2 cells), so part of the OSC gain is a completeness-for-precision trade, not pure
targeting. The column-axis coverage jump (0.73→0.93) is nonetheless a real gain on
fixed vocabulary.

### Relation to H2
This refines H2's causal clause. The spec hypothesized the enumeration *advantage*
grows with depth; what we measure is that header depth is a genuine causal source
of the completeness difficulty for the enumeration method itself — the tree that
should enable scope enumeration instead makes query→header-path resolution harder.
The contribution (re-localizing completeness to header-path decomposition) is thus
causally tied to the hierarchy, not an artifact of dataset choice.

Artifact: `results/e3_depth.json` ·
reproduce: `PYTHONPATH=. python scripts/e3_depth.py --split dev`

## (C) Dense baseline under flattening — depth is a method-specific liability

Same leaf-flatten manipulation, dense baseline (mode="plain", k=10), paired n=158:

| flatten (d→1) effect | OSC original → flat | Δ |
|---|---|---|
| **enumeration** | 0.335 → 0.570 | **+0.234** |
| **dense baseline** | 0.772 → 0.703 | **−0.070** |

**The two methods respond to depth in opposite directions.** Flattening *helps*
enumeration (+0.23) but *hurts* the dense baseline (−0.07): the baseline is
depth-robust — if anything the extra header tokens of a deep tree slightly aid
similarity retrieval, and removing them costs a little.

**Conclusion.** Header depth is **not** an intrinsic property of the
operand-set-completeness problem — the similarity baseline does not suffer from it.
It is a **method-specific liability of the resolve-then-enumerate approach**:
the deterministic fuzzy resolver cannot cleanly map a query to a deep header path,
so the enumerated scope misses operands on deep tables. Combined with W4b (LLM
scale does not fix the decomposer), the single open problem is sharpened to:
**depth-robust query→header-path resolution** — a representation problem, not a
model-capacity or budget problem.

Artifact: `results/e3_depth_dense.json` ·
reproduce: `PYTHONPATH=. python scripts/e3_depth.py --split dev --dense`
