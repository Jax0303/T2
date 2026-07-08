# Embedding tree-node resolver + recall-first (idea follow-up)

Tests the idea: *represent the row-header hierarchy as a tree and match query→header
by semantic embedding (not lexical overlap)* — aimed at the E2/W4b row-axis
decomposition bottleneck. LLM-free. Population: HiTab dev arithmetic m≥2, n=158.

## Resolver comparison (E2 enumeration, paired)

| | lexical (fuzzy) | embed (tree-node) | hybrid (row=embed, col=lexical) |
|---|---|---|---|
| row-axis coverage | 0.544 | 0.582 | **0.582** |
| col-axis coverage | 0.728 | 0.677 | **0.728** |
| OSC enum | 0.335 | 0.361 | **0.380** |
| n decomp correct | 53 | 57 | **60** |
| mean cells | 17.2 | 16.6 | 19.1 |

- **Embedding helps the row axis** (the bottleneck): vocabulary mismatch
  ("building sector" vs "construction") that scored 0 under lexical overlap is
  recovered semantically. It **hurts the column axis** (years/codes like "2014"
  match better lexically).
- **Hybrid** (embed rows, lexical cols) keeps both gains → best OSC 0.380.
- **Equivalence to a 9× larger LLM:** the hybrid (LLM-free) matches the 70b-LLM
  decomposition result exactly (OSC 0.380, n_decomp 60). This nails the
  "representation problem, not model capacity" claim: a targeted representation
  fix equals a frontier-size model at zero LLM cost.
- Still does not beat the dense baseline (ΔOSC vs k=10 = −0.392): the residual
  row-axis failures are **structural** (which/how-many sibling rows to aggregate),
  not vocabulary — embeddings fix lexical/depth, not scope selection.

## Depth re-test (E3 with embed resolver) — the idea's decisive test

Flatten-to-depth-1 effect on enumeration OSC (depth penalty):

| resolver | OSC orig → flat (Δ) | row-cov orig → flat |
|---|---|---|
| lexical | 0.335 → 0.570 (**+0.234**) | 0.544 → 0.601 (flatten helps rows) |
| embed | 0.361 → 0.487 (**+0.127**) | 0.582 → **0.551** (flatten *hurts* rows) |

The embedding resolver **halves** the overall depth penalty and, on the row axis,
**flips depth from a liability to a slight asset** (flattening now lowers row
coverage). Confirms the idea's hypothesis on the bottleneck axis: with semantic
matching, deep row trees are no longer harmful. The residual flatten benefit is now
almost entirely the column axis (numeric/temporal codes).

## E5 — recall-first: can we guarantee 100% completeness, at what cost?

The professor's constraint (operand set must be ~100% complete). Completeness/cost
ladder (OSC = completeness, mean_cells = budget); whole table ≈ 162 numeric cells.

| config | OSC | mean cells | saving vs whole table |
|---|---|---|---|
| A enum precise (hybrid) | 0.380 | 19 | 88% |
| B enum axis-complete | 0.930 | 87 | 46% |
| C dense top-20 | 0.918 | 99 | 39% |
| **D union(B, dense)** | **1.000** | 123 | 24% |
| E whole table | 1.000 | 162 | 0% |

- **100% completeness IS achievable** (union D: every operand on all 158 queries)
  — similarity ranking alone cannot guarantee this; enumeration ∪ dense can.
- **But the cost is high:** 123 cells ≈ 76% of the whole table. "100% completeness"
  is, in practice, "give the model most of the table".
- **Completeness and precision are in tension:** precise (19 cells) → 0.38; complete
  (1.0) → ~whole table. "100% within a *small* set" is unsolved.
- Sweet spot **B**: 0.93 at half the table.

**Open problem (sharpened):** 100%-completeness in a *small precise* set requires
solving **structural scope selection** (which sibling rows, how many) — not lexical
matching (closed by the embedding idea) and not brute-force widening (the union).

Artifacts: `results/e2_osc_enum_embed.json`, `results/e2_osc_enum_hybrid.json`,
`results/e3_depth_embed.json`, `results/e5_recall_first.json` ·
reproduce:
`PYTHONPATH=. python scripts/e2_osc_enum.py --split dev --resolver hybrid` ·
`PYTHONPATH=. python scripts/e3_depth.py --split dev --resolver embed` ·
`PYTHONPATH=. python scripts/e5_recall_first.py --split dev`
