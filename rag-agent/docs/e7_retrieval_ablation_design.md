# E7 — Controlled retrieval ablation at a fixed solver (design)

**Goal.** Prove the contribution is the **retrieval method** (header-tree enumeration
of the operand scope), **not** the generation model. We hold the LLM solver and the
prompt format constant and change **only the retrieved cell set**, then measure
end-to-end answer accuracy. This neutralizes the "you just used a bigger LLM than
HiTab's 2022 BERT parser" objection: every arm uses the *same* LLM, so any accuracy
difference is attributable to retrieval alone.

## The one knob that varies

```
question ──▶ [RETRIEVAL ARM]  ──▶ context (same format) ──▶ [FIXED LLM solver] ──▶ answer
                  ▲ only this changes                          ▲ frozen across arms
```

- **Frozen:** solver LLM + decoding params + prompt template (E4's winning
  `(header-path = value)` structured format) + population + scoring.
- **Varied:** which cells are retrieved into the context.

## Retrieval arms

| arm | what it is | source | role |
|---|---|---|---|
| `dense_topk` | similarity top-k row-chunks (standard table-RAG) | E1 `mode=plain` | **baseline to beat** |
| `enum_base` | header-tree enumeration, hybrid resolver | E6 base | ours (no treatment) |
| `enum_treated` | enumeration + total-row + sibling augmentation | E6 `T_both` | **ours (main)** |
| `whole_table` | dump every numeric cell | — | the "give it everything" point (what worked at home); completeness-free, budget-infeasible |
| `oracle` | gold operand cells exactly | E4 | ceiling (isolates solver limit) |

## Fairness controls (each neutralizes a specific reviewer objection)

1. **Same LLM + same format, all arms.** Objection neutralized: "it's the LLM, not
   retrieval." The LLM's own arithmetic/grounding error (E4 measured silent-wrong
   ≈ 0.42) is *constant* across arms, so it cancels in the paired difference.
2. **Budget matching (critical).** Dense and enumeration return different cell
   counts. We compare **at matched context size**: sweep `dense_topk` over
   k∈{1,3,5,10,20}, record each arm's mean cell count, and plot **answer accuracy
   vs cells**. The claim is "ours sits above the dense accuracy-vs-budget curve" —
   not a single cherry-picked k. Objection neutralized: "you just fed more cells."
3. **OSC reported next to accuracy.** For every arm, report retrieval OSC *and*
   answer accuracy. This shows the causal chain (completeness → correct answer) and
   that accuracy gains come from recovering operands, not prompt luck.
4. **Same population.** HiTab dev arithmetic, distinct-cell scope **m≥2 (n=161)**
   primary; m=1 anchor reported separately. Selection/comparison excluded (gold
   value-matching limitation, already documented).
5. **Paired statistics.** Per-query McNemar + paired bootstrap 95% CI on answer
   accuracy, arm vs `dense_topk` at matched budget.

## Metrics

- **Answer accuracy (primary):** numeric-match (NM) of the parsed answer to gold,
  reusing E4's scorer (non-number rate reported too).
- **Retrieval OSC (mechanism):** operand-set completeness of the arm's cell set.
- **Cells (budget/precision):** mean retrieved numeric cells.

## Hypotheses

- **H5 (main):** at a *fixed* LLM solver and *matched* budget, header-tree
  enumeration (+treatments) gives higher answer accuracy than dense retrieval, and
  ΔAccuracy is explained by ΔOSC (accuracy rises with retrieval completeness).
- **H5a:** `whole_table` reaches high accuracy (the home result) but at ~162 cells;
  `enum_treated` approaches it at a fraction of the budget (precision argument).
- **H5b:** `oracle − enum_treated` accuracy gap = the *retrieval* headroom left;
  `1 − oracle` accuracy = the *solver's own* ceiling (not a retrieval problem).

## What we will and will NOT claim

- ✅ **Claim:** "Holding the generator fixed, replacing similarity retrieval with
  header-tree operand enumeration improves aggregation answer accuracy by Δ at equal
  budget; the gain tracks operand-set completeness." (retrieval contribution, clean)
- ✅ **Claim:** completeness is scope-size-robust (E1 collapse vs ours flat) — the
  failure mode end-to-end accuracy alone hides.
- ⚠️ **Reference only, not a head-to-head win:** HiTab MAPO/TaBERT **45.5% dev** is
  reported as an external reference line (different generator generation); we do
  **not** rest the contribution on beating it, because the model generations differ.
- ❌ **Will not claim:** higher raw OSC than dense at large budget (we lose that;
  E6) — the win is at matched/small budget and on scope-robustness.

## Implementation plan (reuse existing parts)

New script `scripts/e7_retrieval_ablation.py`:
1. population + gold = E6 loader (arithmetic m≥2).
2. per query, per arm → build the retrieved cell set:
   - `dense_topk`: `retrieve(mode="plain", k)` → covered cells (E1/E6 machinery).
   - `enum_*`: `enumerate_scope(...)` with/without treatments (E6).
   - `whole_table` / `oracle`: trivial.
3. render context in E4's `(header-path = value)` format from the cell set.
4. call the **fixed** solver (E4 codegen, `--llm groq:llama-3.1-8b-instant`),
   parse numeric answer, score NM (E4 scorer).
5. record {accuracy, OSC, cells} per arm; paired McNemar + bootstrap CI vs
   `dense_topk`; write `results/e7_retrieval_ablation.json` + summary.

Robustness (optional, if rate limits allow): repeat the solver with a second LLM
(e.g. 70b) to show the retrieval ranking of arms is solver-invariant.

Run (LLM-gated; Groq free-tier rate limits apply):
`PYTHONPATH=. python scripts/e7_retrieval_ablation.py --split dev --llm groq:llama-3.1-8b-instant`
