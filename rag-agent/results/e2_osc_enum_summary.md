# E2 (W4) — Header-tree enumeration vs dense baseline (H2)

Treatment: deterministic header-tree scope enumeration (`rag_agent/retrieve/header_enum.py`).
Baseline: dense single-vector retrieval (`mode="plain"`, bge-small, S2). Paired,
same queries. Population: HiTab dev arithmetic aggregations, **m≥2, n=158**.
seed=42, paired bootstrap 95% CI + McNemar.

## Headline (m≥2, n=158)

| metric | value |
|---|---|
| OSC enumeration | **0.335** (mean 17.2 cells) |
| **OSC \| decomposition correct** | **1.000** (n=53) |
| row-axis coverage | 0.544 |
| col-axis coverage | 0.728 |
| OSC dense baseline k=5 | 0.582 |
| OSC dense baseline k=10 | 0.772 |
| ΔOSC vs k=5 | **−0.247** CI [−0.335, −0.158] |
| ΔOSC vs k=10 | **−0.437** CI [−0.519, −0.348] |

## OSC by scope size m

| m | n | OSC enum | OSC \| decomp | row-cov | mean cells |
|---|---|---|---|---|---|
| 1 | 56 | 0.911 | 1.0 | 0.964 | 10.4 |
| 2 | 110 | 0.364 | 1.0 | 0.564 | 15.6 |
| 3–4 | 26 | 0.192 | 1.0 | 0.462 | 14.7 |
| 5–8 | 15 | 0.200 | 1.0 | 0.400 | 27.8 |
| 9+ | 7 | 0.714 | 1.0 | 0.857 | 29.7 |

## Honest read — H2 is *revised*, not naively confirmed

1. **On raw OSC, enumeration LOSES to the dense baseline** (ΔOSC significantly
   negative at both k=5 and k=10). Dense ranking is more robust to a partly-wrong
   query decomposition because it never hard-commits to a header predicate; a
   missed row predicate in enumeration zeroes that query's OSC outright.

2. **But the mechanism H2 claims is fully validated:** conditional on the
   decomposer resolving both axes, enumeration recovers the complete operand set
   **100% of the time, flat across every scope size** (OSC|decomp = 1.0 at m=2…9+).
   The H1 collapse curve is *eliminated* under enumeration — completeness stops
   depending on scope size m.

3. **OSC_enum (0.335) = decomposition success rate (53/158) exactly.** Enumeration
   converts the operand-set-completeness problem into a **header-path decomposition
   problem**, and localizes the bottleneck to the **row axis** (coverage 0.544 vs
   col 0.728). This matches the prior measured decomposition ceiling.

### Why this is the stronger thesis
The Weller et al. (2508.21038) limit says similarity ranking cannot guarantee an
arbitrary operand *subset* — and H1 shows the baseline's OSC indeed decays with m.
Enumeration removes that combinatorial dependence (OSC|decomp flat = 1.0), moving
the ceiling from a theoretically-hard subset-selection problem to a **separable,
tractable decomposition problem** (specifically row-axis header resolution). The
contribution is the *re-localization of the bottleneck*, not a raw OSC win — and
the path to a raw win is now a single well-defined lever: row-axis decomposition.

## Caveats
- Baseline k=10 uses 10 row-chunks (covers ≥10 rows × all cols), an effectively
  larger cell budget than enumeration's 17 cells — part of the raw-OSC gap is budget.
- m=9+ enum is high (0.71) because those are whole-column sums where the row-axis
  fallback (entire axis) trivially covers gold; not a decomposition success.
- Decomposer here is the deterministic `resolve_against_table`. The obvious next
  lever — LLM-refined `resolve_intent` — was tested in W4b (below).

Artifact: `results/e2_osc_enum.json` ·
reproduce: `PYTHONPATH=. python scripts/e2_osc_enum.py --split dev`

---

## W4b — LLM-refined decomposition (8b vs 70b): bottleneck is model-agnostic

Lever: an LLM picks header paths from the real inventory (`resolve_intent`), meant
to lift row-axis coverage and turn the negative ΔOSC positive.

| metric | deterministic | 8b | 70b |
|---|---|---|---|
| row-axis coverage | 0.544 | 0.506 | **0.595** |
| col-axis coverage | 0.728 | 0.684 | 0.722 |
| OSC enum | 0.335 | 0.285 | **0.380** |
| n decomp correct | 53/158 | 45/158 | **60/158** |
| mean enum cells | 17.2 | 16.0 | **8.9** |
| OSC \| decomp correct | 1.000 | 1.000 | 1.000 |
| ΔOSC vs k=10 | −0.437 | −0.487 | **−0.392** (CI [−0.475, −0.310]) |

Refinement reach: 8b refined 103/214 queries, 70b refined 212/214.

**Read — the bottleneck is largely model-agnostic.**
- A *weak* 8b LLM **degrades** decomposition below the deterministic fuzzy ranker
  (worse header paths on the queries it touched).
- A *strong* 70b LLM **partly lifts** it (row-axis 0.544→0.595, n-correct 53→60)
  and is far more precise (mean 17→9 cells), but a ~9× larger model moves row-axis
  coverage by only +0.05 and **still does not beat the dense baseline** (ΔOSC
  significantly negative at both budgets).
- By scope: 70b improves m=2 (0.36→0.45) and m=3–4 (0.19→0.42), but m≥5 drops to
  0.0 — its precise predicates lose the deterministic "dump the whole axis"
  fallback that accidentally covered whole-column sums.
- The enumeration invariant (OSC | decomp = 1.0) holds across all three models.

The row-axis header-decomposition ceiling is **not closed by LLM scale** within the
available range: it is a representation/matching problem, not a model-capacity one.
The next lever is the decomposer's *representation* (structure-aware row-path
matching), not a bigger model.

Artifacts: `results/e2_osc_enum_llm.json` (8b), `results/e2_osc_enum_llm70b.json` (70b) ·
reproduce: `PYTHONPATH=. python scripts/e2_osc_enum.py --split dev --llm groq:llama-3.3-70b-versatile`
