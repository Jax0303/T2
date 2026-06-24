# Operand-Set Completeness Retrieval — Results (W7)

Consolidated results for the operand-set completeness (OSC) study on hierarchical
tables. All numbers measured on HiTab `dev`, seed=42, paired bootstrap 95% CI.
Retrieval experiments (E1, E2, W4b) are LLM-free except W4b's decomposition step.

> Status: E1 (H1) ✅ · E2 (H2) ✅ · W4b (LLM decomposition lever, 8b+70b) ✅ ·
> E3 (synthetic depth) ✅ · E4 (generation format) ✅.

## Evaluation population

Gold operands are resolved from HiTab `linked_cells.quantity_link` by value-matching
into data space (`rag_agent/bench/hitab.py:resolve_gold_operands`). Value-matching
yields a clean operand set **only for arithmetic aggregations** (the answer is a
computed number). For selection/comparison queries (argmax/argmin/greater_than…)
the answer is a *header label*, so there is no value to match and the gold operand
set is empty — these are reported as a limitation, not evaluated.

| population | n | use |
|---|---|---|
| arithmetic, operands resolved | 214 | E1/E2 curve (incl. m=1 anchor) |
| **arithmetic, m≥2 (true scope)** | **158** | **primary OSC population** |
| selection/comparison | (excluded) | value-matching cannot build gold (limitation) |

Integrity: of 1671 dev queries, 22% have empty operand sets — diagnosed as
label-answer queries (349) + 21 genuine value-resolution failures, **not** a
retriever/LLM failure (`results/operand_gold_report.json`).

## Metric

**OSC** (Operand-Set Completeness) = fraction of queries where *every* gold operand
cell is retrieved (all-or-nothing subset containment). Necessary condition for a
correct aggregation answer; strictly harder than averaged per-cell recall.
Implementation + unit tests: `rag_agent/eval/operand_set.py`,
`tests/test_operand_set.py` (10/10).

---

## H1 — dense baseline OSC collapses with scope size (E1)

Baseline: dense single-vector retrieval (`mode="plain"`, bge-small, S2 row-chunks).

**OSC vs scope size m, at fixed budget k** (the collapse):

| k ＼ m | 1 | 2 | 3–4 | 5–8 | 9+ |
|---|---|---|---|---|---|
| 1 | 0.68 | 0.20 | 0.19 | 0.13 | 0.14 |
| 5 | 0.89 | 0.60 | 0.62 | 0.53 | 0.29 |
| 10 | 0.96 | 0.79 | 0.85 | 0.67 | 0.43 |
| 20 | 0.98 | 0.92 | 0.92 | 0.87 | 1.00\* |

\*n=7. **m≥2 (n=158) overall OSC:** k=1 → 0.19, k=5 → 0.58, k=10 → 0.77, k=20 → 0.92.

**Verdict: H1 supported.** At realistic budgets (k≤10) OSC falls monotonically as
the aggregation scope grows. A larger budget partially rescues completeness, but
only by dumping ~20 chunks into context — completeness is bought with budget, not
targeting. First-order independence (r^m) is a reasonable fit at tight budget.
Detail: `results/e1_osc_baseline_summary.md`, `results/e1_osc_baseline.json`.

---

## H2 — header-tree enumeration re-localizes the bottleneck (E2)

Treatment: deterministic header-tree scope enumeration — resolve the query to
header-path predicates, then enumerate every numeric leaf under the matched scope
nodes (`rag_agent/retrieve/header_enum.py`, tests 4/4). Paired vs dense baseline,
m≥2, n=158.

| metric | value |
|---|---|
| OSC enumeration | 0.335 (mean 17.2 cells) |
| **OSC \| decomposition correct** | **1.000** (n=53) |
| row-axis coverage | 0.544 |
| col-axis coverage | 0.728 |
| ΔOSC vs dense k=5 | −0.247, CI [−0.335, −0.158] |
| ΔOSC vs dense k=10 | −0.437, CI [−0.519, −0.348] |

OSC | decomposition-correct, **by scope size**: 1.0 at m=2, 3–4, 5–8, 9+ — flat.

**Verdict: H2 revised, not naively confirmed.**
1. On *raw* OSC, enumeration **loses** to the dense baseline (ΔOSC significantly
   negative): a missed header predicate zeroes a query, whereas similarity ranking
   degrades gracefully.
2. But the **mechanism is fully validated**: conditional on correct decomposition,
   enumeration recovers the complete operand set **100% of the time, independent of
   scope size**. The H1 collapse curve is *eliminated*.
3. `OSC_enum (0.335) = decomposition success rate (53/158)` exactly. Enumeration
   **converts operand-set completeness into a header-path decomposition problem**
   and localizes the bottleneck to the **row axis** (0.544 vs col 0.728).

The contribution is the **re-localization of the bottleneck** — from the
theoretically-hard arbitrary-subset-selection limit (Weller et al. 2508.21038,
which H1 exhibits) to a separable, measurable decomposition problem — not a raw
OSC win. Detail: `results/e2_osc_enum_summary.md`, `results/e2_osc_enum.json`.

---

## W4b — the decomposition bottleneck is largely model-agnostic

Lever: refine decomposition with an LLM choosing header paths from the real
inventory (`resolve_intent`), to raise row-axis coverage. Tested at two scales.

| metric | deterministic | 8b | 70b |
|---|---|---|---|
| row-axis coverage | 0.544 | 0.506 | **0.595** |
| OSC enum | 0.335 | 0.285 | **0.380** |
| n decomp correct | 53/158 | 45/158 | **60/158** |
| mean enum cells | 17.2 | 16.0 | **8.9** |
| OSC \| decomp correct | 1.000 | 1.000 | 1.000 |
| ΔOSC vs k=10 | −0.437 | −0.487 | **−0.392** |

- A *weak* 8b model **degrades** decomposition below the deterministic fuzzy ranker.
- A *strong* 70b model **partly lifts** it (row-axis +0.05, n-correct 53→60) and is
  far more precise (17→9 cells), but a ~9× larger model still **does not beat the
  dense baseline** (ΔOSC significantly negative).

The row-axis ceiling is **not closed by LLM scale** in the available range — it is a
representation/matching problem, not a model-capacity one. The next lever is the
decomposer's representation, not a bigger model. The enumeration invariant
(OSC | decomp = 1.0) holds across all three. Detail: `results/e2_osc_enum_summary.md`.

---

## E3 — header depth is a method-specific liability (causal)

Holding data, leaf vocabulary, and scope fixed, leaf-flatten every table to depth 1
(drop ancestor header levels) and re-measure, paired (n=158, m≥2, LLM-free).

| flatten (d→1) effect | OSC original → flat | Δ |
|---|---|---|
| enumeration | 0.335 → 0.570 | **+0.234** |
| dense baseline | 0.772 → 0.703 | **−0.070** |

Removing the header tree (same words, same data) **raises enumeration OSC by +0.23**
(col-axis coverage 0.73→0.93) but **lowers the dense baseline by −0.07**. The two
methods respond to depth in *opposite* directions: depth is not intrinsic to the
completeness problem (the baseline is depth-robust) — it is a **method-specific
liability of resolve-then-enumerate**, because the fuzzy resolver cannot map queries
onto deep header paths. Caveat: the flattened enum scope is 2.2× larger (37.9 vs
17.2 cells), so part of its OSC gain trades precision for completeness. Detail:
`results/e3_depth_summary.md`, `results/e3_depth.json`, `results/e3_depth_dense.json`.

Together with W4b, the open problem is sharpened to **depth-robust
query→header-path resolution** — a representation problem, not model scale or budget.

## E4 — structured context cuts silent grounding errors (H3)

Retrieval held fixed at the oracle operand set; only the context *format* varies
(same numbers, same header words). Codegen, Groq llama-3.1-8b, n=158, m≥2.

| arm | NM accuracy | silent-wrong rate |
|---|---|---|
| flat dump | 0.335 | 0.665 |
| **(header-path = value)** | **0.576** | 0.424 |

ΔNM = **+0.241** CI [0.158, 0.323]; McNemar 49:11. **H3 supported** — making the
binding explicit nearly doubles numeric-match accuracy and cuts the silent-error
rate, even with perfect retrieval. Residual silent-wrong 0.42 is the 8b model's
own grounding/arithmetic limit (non-number rate 0). Detail:
`results/e4_format_summary.md`, `results/e4_format.json`.

## Idea follow-up — embedding tree-node resolver + recall-first

Tests the idea "represent row headers as a tree and match query→header by semantic
embedding" against the row-axis bottleneck. LLM-free, n=158.

| resolver | row-cov | col-cov | OSC enum |
|---|---|---|---|
| lexical (fuzzy) | 0.544 | 0.728 | 0.335 |
| embed (tree-node) | 0.582 | 0.677 | 0.361 |
| **hybrid (row=embed, col=lexical)** | 0.582 | 0.728 | **0.380** |

- Embedding fixes the **row axis** (vocabulary mismatch) but hurts the **column
  axis** (years/codes match better lexically) → hybrid keeps both. The hybrid is
  LLM-free yet **matches the 70b LLM exactly** (OSC 0.380, n_decomp 60): a targeted
  representation fix equals a 9× larger model — nailing "representation, not scale".
- **Depth re-test (E3 embed):** the embedding resolver halves the flatten benefit
  (+0.234→+0.127) and on the row axis flips depth from liability to slight asset
  (row-cov 0.582→0.551 when flattened). The idea's hypothesis holds on the
  bottleneck axis.
- Still below the dense baseline: the residual is **structural scope selection**
  (which/how-many sibling rows), not vocabulary.

**Recall-first (E5) — meeting a 100%-completeness requirement:**

| config | OSC | mean cells (whole table ≈162) |
|---|---|---|
| enum precise (hybrid) | 0.380 | 19 |
| enum axis-complete | 0.930 | 87 |
| dense top-20 | 0.918 | 99 |
| **union(axis-complete, dense)** | **1.000** | 123 |
| whole table | 1.000 | 162 |

100% completeness **is** achievable (union), which similarity ranking alone cannot
guarantee — but it costs ~76% of the whole table. Completeness and precision are in
tension; "100% within a *small* set" remains the open problem. Detail:
`results/e2e3_embed_resolver_summary.md`.

## Differentiation gate (W0)

All four nearest works verified (method sections, `docs/RELATED_DELTA.md`):
DCTR and Huawei-TableRAG **exclude aggregation from retrieval** and defer it to SQL
on flat relational schemas; T-RAG's "hierarchical" is a corpus index (its benchmark
lacks operand labels); HD-RAG models the internal header tree but only for top-1
*document* retrieval, never enumerating the scope at retrieval time. **No prior work
puts header-tree scope enumeration / operand-set completeness as a retrieval-time
objective.** Gate passes.

## Hypothesis scorecard (numbers only)

| H | claim | verdict |
|---|---|---|
| H1 | dense single-vector OSC degrades with scope size m | **supported** (E1) |
| H2 | header-tree enumeration improves operand-set completeness | **revised**: removes scope-size dependence (OSC\|decomp=1.0 flat) and re-localizes the bottleneck to row-axis decomposition; does **not** beat raw baseline OSC under the deterministic/8b/70b decomposer (E2, W4b) |
| H2-causal | the enumeration effect is hierarchy-caused, not domain | **supported, with a twist** (E3): depth causally suppresses enumeration OSC (flatten→ +0.234) but the dense baseline is depth-robust (−0.070) — depth is a method-specific liability of resolve-then-enumerate, not intrinsic to completeness |
| H3 | structured (header-path, value) context reduces silent grounding errors | **supported** (E4): oracle-fixed retrieval, ΔNM +0.241 [0.158, 0.323], silent-wrong 0.66→0.42 |

## Threats / limitations

- Single dataset (HiTab); external validity to finance/aviation hierarchies untested.
- OSC is upper-bounded by header-path decomposition accuracy; we report it
  conditionally (OSC | decomp) to separate enumeration from decomposer quality.
- Selection/comparison aggregations excluded from gold (value-matching limitation).
- Baseline k=10 uses a larger effective cell budget than enumeration's ~17 cells;
  part of the raw-OSC gap is budget, not targeting.
- W4b tested 8b and 70b; LLM scale does not close the row-axis ceiling, but only
  Groq-hosted Llama models were tried (no frontier model / no fine-tuned decomposer).

## Reproduce

```
PYTHONPATH=. python scripts/build_operand_gold.py --split dev
PYTHONPATH=. python scripts/e1_osc_baseline.py   --split dev
PYTHONPATH=. python scripts/e2_osc_enum.py       --split dev
PYTHONPATH=. python scripts/e2_osc_enum.py       --split dev --llm groq:llama-3.1-8b-instant
PYTHONPATH=. python scripts/e3_depth.py          --split dev [--dense]
PYTHONPATH=. python scripts/e4_format.py         --split dev --llm groq:llama-3.1-8b-instant
```
