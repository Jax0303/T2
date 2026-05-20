# Adaptive Table-RAG: routing between original 2-D structure and a vector DB

Master's-thesis project on **how to combine a vector retrieval index with the
original 2-D table structure** so a free LLM can answer hard HiTab queries
(formulas / functions from the paper appendix) reliably.

The starting question was:

> *"For a given hard table-QA query, **when** should the agent trust the
> vector DB, **when** should it fall back to the original 2-D table, and
> **when** should the LLM not be doing the math at all?"*

Earlier work in this repo (HART, §5) tried to **blend** the two signals with
a single score (α-weighted cosine + header-alignment) — the result was
negative. This project takes the opposite approach: keep the two stores
strictly separate and **route the query through different stages** depending
on what it asks for.

---

## Architecture

```
                                  query
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │  (1) Query intent classifier            │
              │      rule-based, 6 labels mirroring     │
              │      HiTab appendix supervision         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │  (2) Stage policy                       │
              │      decides which of the stages below  │
              │      actually run for THIS query        │
              └────────────────────┬────────────────────┘
                                   │
            ┌──── reasoning_only ──┴────── everything else ────┐
            │                                                  │
            ▼                                                  ▼
   ┌────────────────┐                  ┌──────────────────────────────────┐
   │ LLM alone      │                  │  (3) Vector retrieval            │
   │ (skip both DBs)│                  │      Chroma + bge-large-en-v1.5  │
   └────────────────┘                  │      top-20 vectors → top-5 tabs │
                                       └──────────────┬───────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────────┐
                                       │  (4) Verifier — *original DB*    │
                                       │      keyword overlap (q ↔ headers)│
                                       │      number overlap  (q ↔ cells)  │
                                       │      rerank: 0.7·vec + 0.3·verify │
                                       └──────────────┬───────────────────┘
                                                      │
                       ┌── arithmetic / multi-op ─────┴──── lookup / arg / cmp ──┐
                       │                                                          │
                       ▼                                                          ▼
        ┌──────────────────────────────────┐                        ┌──────────────────────┐
        │  (5a) Symbolic compute           │                        │  (5b) LLM reader     │
        │       LLM emits JSON             │                        │       reads verified │
        │       {cells, expression}        │                        │       top-1 table,   │
        │       → header-path resolve      │                        │       returns        │
        │       → safe AST eval (no eval())│                        │       "Final answer" │
        │  Gate: adopt only if ≥2 ops or   │                        └──────────┬───────────┘
        │  strong arithmetic intent        │                                   │
        └──────────────┬───────────────────┘                                   │
                       │                                                       │
                       └──────────────────────────┬────────────────────────────┘
                                                  ▼
                                              ANSWER
                                       (with full per-stage trace)
```

**Two DBs, two roles:**

| Store | Role | What it answers |
|---|---|---|
| Vector DB (Chroma + bge-large) | Candidate discovery | "Which 5 tables might be relevant?" |
| Original 2-D DB (parsed HiTab JSON + header tree) | Verification + arithmetic | "Does this candidate actually contain the entities/numbers the query mentions? What is the exact cell value at (`row_header="total"`, `col_header="2017 actual"`)?" |

The LLM is used in **two narrowly scoped roles**: cell-extractor (JSON
emitter for arithmetic) and reader (natural-language answer for lookup /
arg / comparison classes). It never does the arithmetic itself.

### Detailed data flow (what runs and what is produced)

Same pipeline as above, but annotated with the function call that runs
each step, the data structure it produces, and what file holds it.

```
 query : str
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (1) classify_query(q)                          router/query_classifier│
│      regex over 6 patterns (math syms ≥2, arith triggers, entity-cue, │
│      arg/pair, comparison, total-as-aggregation)                      │
│      → QueryIntent(qtype, needs_table, needs_symbolic, signals)       │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (2) plan_stages(intent)                                    router/policy│
│      → Plan(stages=[RETRIEVE,VERIFY,(SYMBOLIC,)LLM_ANSWER], reason)    │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (3) VectorStore.search(q, top_k_vectors=20, top_k_tables=5)           │
│        stores/vector_store.py                                         │
│    a) embed q  : bge-large-en-v1.5 → 1024-d vector                    │
│    b) chroma   : collection.query(emb, n_results=20) → 20 chunk hits  │
│    c) per table: dedup by table_id, keep best score per table         │
│    → List[VectorHit(table_id, score, vector_id, chunk_text)]          │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (4) rerank(q, hits, original_store, 0.7, 0.3)         retrieve/verifier│
│    for each of the 5 candidate hits:                                  │
│      table = original_store.get(hit.table_id)   ← FIRST use of orig.  │
│      kw_overlap  = |query_kw ∩ table_header_kw| / |query_kw|          │
│      num_overlap = |query_nums ∩ table_cell_nums| / |query_nums|      │
│      verify_conf = 0.6·kw + 0.4·num   (or kw alone if no nums)        │
│      final_score = 0.7·hit.score + 0.3·verify_conf                    │
│    sort by final_score → top-1 = top_table                            │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
       ┌────── arithmetic intent (SYMBOLIC in plan) ──────┐
       │                                                   │
       ▼                                                   ▼
┌────────────────────────────────────────┐    (skip 5a, go to 5b)
│ (5a-i)  extract_plan(llm, q, top_table)│
│          extract/cell_extractor.py     │
│    render table → text                 │
│    system prompt: emit JSON only       │
│    user: table + question              │
│    parse JSON {cells, expression}      │
│    → ExtractedPlan(cells, expression)  │
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ (5a-ii) evaluate_plan(plan, top_table) │
│          extract/symbolic_eval.py      │
│    for each cell:                      │
│      OriginalTable.resolve(rh, ch)     │
│        → word-bounded token match      │
│           on joined " :: " path        │
│        → (row, col, value)             │
│      env[var] = float(value)           │
│    ast.parse(expression, "eval")       │
│    walk tree with whitelist:           │
│      Constant, Name, BinOp(+-*/),      │
│      UnaryOp(+,-) only                 │
│    → SymbolicResult(ok, value, ...)    │
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ (5a-iii) adoption gate    agent.py     │
│   op_count = count "+-*/" in expr      │
│   adopt = sym.ok AND (                 │
│     op_count >= 2                      │
│     OR (intent==arith AND ops>=1       │
│         AND cells>=2))                 │
│   if adopt: answer = sym.value         │
└──────────────────┬─────────────────────┘
                   │ (not adopted → fall through to 5b)
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (5b) LLM reader                                            agent.py   │
│    render top_table as text (title + header paths + data rows)        │
│    system: "Reasoning: ... Final answer: ..."                         │
│    LLM.complete(system, user)                                         │
│    regex extract "Final answer: (.+)"                                 │
│    → answer string                                                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (6) AgentResult                                            agent.py   │
│    query, intent, plan,                                               │
│    vector_ranked[5], final_ranked[5], top_table_id,                   │
│    symbolic (plan + resolved_cells + AST value + adopted flag),       │
│    reader (raw output + parsed answer),                               │
│    answer, source ("symbolic"|"reader"), elapsed_s                    │
│    → JSON-serialised for offline metric re-derivation                 │
└──────────────────────────────────────────────────────────────────────┘
```

The flow guarantees that **`Python's eval() is never called**. The cell
extractor's JSON output is parsed as data, the expression is walked as an
AST with a whitelist of node types, and any `Call` / `Attribute` /
`Import` node aborts with `ValueError`. Tested with
`__import__("os").system("…")` payloads.

---

## Hypotheses and results

Tested on a stratified **40-query hard subset** of HiTab dev (8 per
difficulty class, derived from HiTab's appendix supervision: `aggregation`
array + `answer_formulas` op count). Seed = 0.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Using the original DB only to **verify** vector hits (not to retrieve) lifts R@1 | ✅ confirmed | R@1: 0.575 → **0.675** (+10 pp). R@5 = 0.875. |
| H2 | Different query classes need different stages, not one monolithic pipeline | ✅ confirmed | Entity-answer classes (pair / single_arg / comparison): NM 0.50 – 0.875. Arithmetic with reader alone: 0.125. Forcing the same pipeline for both hurts. |
| H3 | Arithmetic answers should come from deterministic compute, not the LLM | ✅ partial | `comparison_or_count` symbolic exec accuracy = **0.375** (3/8 fully deterministic). `multi_op_formula` symbolic still 0% — cell-selection is the real ceiling, not arithmetic. |
| H4 | Among free LLMs, the reader choice matters more than retrieval algorithm tuning | ✅ confirmed | Same retrieval + verifier + symbolic. Reader = Llama-3.1-8B (Groq): NM 0.150. Reader = Qwen-2.5-7B 4-bit (local): NM **0.450** — 3× higher. |

### Final headline (Qwen-2.5-7B reader, **v3.1** — all 4 audit bugs fixed)

| Metric | Value | 95% CI (paired bootstrap, n=40) |
|---|---:|---|
| R@1 (vector only) | 0.575 | [0.425, 0.725] |
| **R@1 (after verifier)** | **0.675** | [0.525, 0.825] |
| R@5 | 0.875 | [0.775, 0.975] |
| Exact Match | 0.325 | [0.175, 0.475] |
| **Numeric Match** | **0.475** | **[0.325, 0.625]** |
| **Δ R@1 (verifier, paired)** | **+0.100** | [0.000, 0.225] |

Compared to the existing hard-query baseline (Sidecar + CoT, **NM = 0.250**),
v3.1's CI lower bound (0.325) sits above the baseline — the +22 pp gain is
statistically meaningful at this sample size.

### Audit runs (lab-meeting bullet-proofing)

| Run | NM | Δ vs v3.1 | what it tests |
|---|---:|---:|---|
| **v3.1 (final, seed=0)** | **0.475** | — | all four audit bugs fixed |
| Verifier ablation (`w_verify=0`) | 0.350 | −12.5 pp | "is the verifier really doing the work?" → **yes**, paired Δ R@1 +10 pp [0, 0.225] |
| seed = 1 | 0.400 | −7.5 pp | stability — not cherry-picked, mean across 3 seeds = 0.417 |
| seed = 2 | 0.375 | −10.0 pp | |
| Qwen reader + **Groq Llama-3.3-70B as cell-extractor** | 0.455 (n=33) | — | arithmetic_agg NM **0.125 → 0.375 (×3)**, comparison_or_count **0.750 → 1.000**, multi_op_formula still 0 — the 70B extractor helps arithmetic but does *not* rescue multi-cell selection |

Honest trade-off found in ablation: the verifier *helps* on average but
*hurts* multi_op_formula R@1 by −12.5 pp (these queries have low keyword
overlap with their table, so the verifier's keyword signal pushes the
wrong table up). A query-class-aware verifier weight is the natural fix.

Full per-class numbers, the audit-bug-progression, and the failure-case
trace in [`rag-agent/EXPERIMENTS.md`](rag-agent/EXPERIMENTS.md).

---

## What an actual hard query looks like

The 5 difficulty classes come from the HiTab paper's appendix (derived
from each sample's `aggregation` array + Excel-style `answer_formulas`).
One real example per class, all from the v3.1 run on HiTab dev:

### `multi_op_formula` — Excel formula with ≥ 2 arithmetic ops

```
Q: "what is the percentage of southern asia, southeast asia and east asia
    consisting of economic immigrants?"
HiTab gold formula:  =B20+B21+B22
HiTab gold answer:   55.8
```

The agent's full trace on this query:

```
intent  : arithmetic_agg → run [retrieve, verify, symbolic, llm_answer]
retrieve: vector top-5 = [2793, 2581, 208, 2658, 755]
verify  : rerank top-5  = [2793, 208, 2581, 755, 2658]   ← gold = 2793, lifted to #1
symbolic: LLM emitted
    {"cells": [
       {"var":"x1","row":"percent > source region > southern asia", "col":"economic class"},
       {"var":"x2","row":"percent > source region > southeast asia","col":"economic class"},
       {"var":"x3","row":"percent > source region > east asia",     "col":"economic class"}],
     "expression":"(x1 + x2 + x3)"}
    resolved via header-path lookup → (18.7) + (15.4) + (21.7) = 55.8
final answer: 55.8  ✓ matches gold
```

This is exactly the case that **bug #4 (word-boundary resolver)** was
fixing: in v3 the substring `"east asia"` collapsed onto the row
`"southeast asia"`, and the agent computed 18.7+15.4+15.4 = 49.5.

### `arithmetic_agg` — single aggregation (sum/diff/avg/range/div)

```
Q: "what is the range of the largest difference outside quebec related to
    the perception of the rcmp as a very important national symbol?"
HiTab gold formula:  =MAX(E11:E15,E5:E9)
HiTab gold answer:   [78, 54]
predicted:           "24"   (NM = False, this one fails)
```

`MAX(...)` over disjoint ranges is hard for both reader and symbolic
extractor; the LLM picked the wrong range.

### `pair_or_topk_arg` — pair-argmax / argmin / top-k pick

```
Q: "which is more likely to report having a large number of close friends,
    senior men living in couples or senior men living alone?"
HiTab gold formula:  =E4   (i.e. one entity name from the headers)
HiTab gold answer:   ["living in a couple"]
predicted:           "living in a couple"   ✓
```

Answer is an **entity name**, not a number. The LLM reader handles this
directly; symbolic is skipped by the policy (no arithmetic intent).

### `single_arg` — argmax / argmin / max / min over one column

```
Q: "which area had the least homelessness support workers among ontario,
    british columbia and quebec?"
HiTab gold formula:  =A11
HiTab gold answer:   ["quebec"]
predicted:           "Quebec"   ✓
```

### `comparison_or_count` — greater/less / opposite / counta

```
Q: "how many percentage points does intra-provincial trade fall due to
    reduced border costs?"
HiTab gold formula:  =-E6        ← "opposite" — gold is the magnitude (46.1),
                                   the value in the cell is the signed -46.1
HiTab gold answer:   46.1
predicted:           "-46.1"     ✓ (NM matches via the abs() variant)
```

`Numeric Match` accepts the sign-flipped form because HiTab's `opposite`
aggregation defines this convention.

### A second symbolic-route success (`comparison_or_count`)

```
Q: "what's the percent that mfp without utilization adjustment declined
    over the period from 2000 to 2009?"
HiTab gold formula:  =-(C5)
HiTab gold answer:   0.9
predicted (symbolic): -0.9
  resolved x1 = 0.3   (percent > mfp growth, period column)
  expression: x1
```

Routed through the symbolic path; the negative sign comes from the
`=-(...)` form and is matched by NM's abs() variant. **Three out of eight
comparison_or_count queries are answered this way** — fully deterministic,
no LLM arithmetic.

---

## What each metric means (with v3.1 actual numbers)

| Metric | v3.1 value | What it answers | What this number means here |
|---|---:|---|---|
| **R@1 (vector only)** | 0.575 | After pure embedding search, what fraction of queries return the gold table as the #1 candidate? | 23 of 40 queries land the right table on first hit using cosine similarity alone — a reasonable baseline for bge-large on HiTab. |
| **R@1 (after verifier)** | **0.675** | After cross-checking the top-5 against the original 2-D structure (keyword + numeric overlap) and reranking, does gold reach #1? | 27 of 40. **The verifier promotes 4 extra queries from #2 / #3 to #1.** Paired 95% CI on the +10 pp delta: [0.000, 0.225] — borderline at two-sided, p < 0.025 one-sided. |
| **R@5** | 0.875 | Is the gold table somewhere in the top 5 (so the reader still has a chance)? | 35 of 40. The remaining 5 queries are unrecoverable by the LLM regardless of how good it is — retrieval missed entirely. **R@5 is the ceiling that any downstream LLM can possibly hit.** |
| **MRR** | 0.759 | Mean reciprocal rank — average of (1 / position of gold). 1.0 = always #1, 0.5 = always #2, etc. | Average rank is ~1.3 — gold is usually at #1, occasionally at #2. Mid-rank failures are rare. |
| **nDCG@10** | 0.789 | Position-weighted relevance, log₂ discount. 1.0 = always #1. | Confirms MRR — most ranks are very near the top; the tail isn't dragging the score. |
| **Exact Match (EM)** | 0.325 | Does the predicted string equal a gold-list element *after* lower-case strip? | 13 of 40. Most failures are formatting: predicted `"-46.1"` vs gold `46.1`, or "Quebec" vs `["quebec"]`. EM punishes the agent for surface-level differences that don't change the answer. |
| **Numeric Match (NM)** | **0.475** | The HiTab paper's tolerant matcher. ±2 % rel-tol on numbers, accepting ×100 / ÷100 / abs() variants (for percent / fraction / opposite); case-insensitive substring for string gold. | **19 of 40 queries answered correctly.** This is the headline figure and it is +22.5 pp above the existing hard-query bench (NM = 0.250). 95 % CI [0.325, 0.625] — lower bound sits above the baseline → significant at n=40. |
| **sym_attempted** | 0.300 | What fraction of queries did the LLM produce a JSON cell-extraction plan that the AST evaluator could actually compute? | 12 of 40. Symbolic only fires on arithmetic-intent classes; the gate filters out trivial 1-op extractions on non-arithmetic queries. |
| **sym_correct** | 0.125 | Of the queries where symbolic fired, how many produced a number that matches gold under NM? | 5 of 40 are answered **entirely without LLM arithmetic** — pure header-path lookup + AST eval. Concentrated in `comparison_or_count` (3/8 = 0.375). |

### How to read the trade-offs

- **Vector vs verifier on multi_op_formula**: R@1 vector = 0.625, R@1
  final = 0.500. The verifier **demotes the gold table** for one
  multi-op query (8 → 7 of 8 lost). Reason: multi-op questions tend to
  use generic words ("total", "percentage", "sum") that match many
  tables' headers, so the verifier's keyword signal pushes a similarly-
  worded but wrong table to #1. This is a known weakness — query-class
  aware weights would fix it.
- **EM 0.325 vs NM 0.475**: 6 queries are scored as wrong by EM but
  correct by NM. All 6 are `pred="-46.1"` style: sign flip due to
  HiTab's `opposite` aggregation, where the answer is the magnitude and
  the cell is negative. Both metrics agree on whether the agent
  *understood* the question; they disagree on whether to count it as
  "correct" — NM follows the HiTab paper's convention, EM is the
  literal-string baseline.
- **sym_correct concentrated in `comparison_or_count`**: this class
  often only needs one cell (`=-(C5)`, `=C7-C9`) — easy to extract.
  `multi_op_formula` needs 3 – 6 cells from specific rows. Even after
  the resolver fix the LLM picks the right cells only 1 of 8 times,
  which is the genuine bottleneck.

---

## Per-class breakdown (v3.1 final)

| Class | n | R@1 (vec) | **R@1 (final)** | R@5 | MRR | nDCG | EM | **NM** | sym_correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_op_formula | 8 | 0.625 | 0.500 | 0.875 | 0.688 | 0.737 | 0.125 | **0.125** | 0.125 |
| arithmetic_agg | 8 | 0.375 | 0.375 | 0.750 | 0.504 | 0.565 | 0.125 | **0.125** | 0.125 |
| pair_or_topk_arg | 8 | 0.500 | **0.875** | 1.000 | 0.938 | 0.954 | 0.750 | **0.875** | 0.000 |
| single_arg | 8 | 0.625 | 0.750 | 0.750 | 0.750 | 0.750 | 0.500 | **0.500** | 0.000 |
| comparison_or_count | 8 | 0.750 | **0.875** | 1.000 | 0.917 | 0.938 | 0.125 | **0.750** | **0.375** |
| **OVERALL** | 40 | 0.575 | **0.675** | **0.875** | 0.759 | 0.789 | 0.325 | **0.475** | 0.125 |

---

## What the audit revealed (and why per-query traces matter)

The first run reported `multi_op_formula NM = 0.000` and we initially read
it as "readers can't do multi-cell arithmetic" — consistent with prior
literature. Reading the per-query traces flipped that story:

1. **Header-separator bug**: the cell-extractor prompt rendered columns as
   `"A > B > C"`, the resolver joined with `" :: "` and did a single
   substring check. **Every** symbolic extraction failed with
   `unresolved_cell`. The 0.000 was a measurement artifact.
2. **Classifier misroute**: "who/which had higher proportion of …?" was
   routed to `arithmetic_agg` because of the word *proportion*, and a
   spurious `x1 - x2 = 8.4` overwrote the reader's correct name-answer.
3. **Symbolic over-firing**: any successful 1-op extraction was being
   adopted even on non-arithmetic queries. Added an op-count gate.

After the three fixes, `multi_op_formula = 0.000` is **real**: the LLM
now extracts plausible plans (3 / 8 produce numbers via AST eval) but
**picks the wrong cells every time**. This relocates the bottleneck:
the limitation is the LLM's *cell-selection* ability, not its arithmetic.
A stronger extractor (Groq 70B partial run showed signal) is the
natural next experiment.

This is the lab-meeting takeaway:

> *Separating the two stores is not a code-organization choice; it is a
> measurement design choice. Each store gives an independent signal, and
> the rerank confidence + symbolic exec accuracy + reader answer can
> be inspected per query to figure out which component is failing. That
> is how the "0% is real" / "0% is a bug" distinction was made.*

---

## Repository layout

The repo is now a single package — earlier exploratory thesis modules
(§3 serialization audit, §4 layer probing, §5 HART retrieval) have been
removed; only the negative-result motivation for §5 is preserved here in
the README and the Sidecar+CoT baseline JSON (`rag-agent/results/
baselines/sidecar_cot_baseline.json`) is kept for the head-to-head
comparison.

```
.
├── README.md                                      this file
├── rag-agent/
│   ├── README.md                                  package overview
│   ├── EXPERIMENTS.md                             full experiment report
│   ├── 발표스크립트.md                              Korean lab-meeting talk script
│   ├── pyproject.toml
│   ├── rag_agent/
│   │   ├── agent.py                               5-stage orchestrator
│   │   ├── data/loader.py                         HiTab JSON loader
│   │   ├── stores/                                OriginalStore + VectorStore
│   │   ├── router/                                query classifier + stage policy
│   │   ├── retrieve/verifier.py                   keyword + numeric overlap rerank
│   │   ├── extract/                               JSON cell extractor + safe AST eval
│   │   ├── llm/                                   Groq + local Qwen backends
│   │   └── eval/metrics.py                        R@k, MRR, nDCG, EM, NM
│   ├── scripts/
│   │   ├── run_eval.py                            benchmark entry point
│   │   ├── smoke_test.py                          offline pipeline smoke
│   │   ├── bootstrap_ci.py                        95 % CI on the headline metrics
│   │   └── aggregate_runs.py                      compare runs side-by-side
│   └── results/
│       ├── baselines/sidecar_cot_baseline.json    prior bench (NM = 0.250)
│       ├── local_qwen7b{,_v2,_v3}.json            run-by-run progression
│       ├── qwen7b_v3.1_resolverfix.json           final (NM 0.475)
│       ├── qwen7b_ablation_noverify.json          verifier OFF
│       ├── qwen7b_seed{1,2}.json                  multi-seed stability
│       ├── qwen7b_groq70b_extractor.json          stronger extractor
│       └── groq_llama3.1_8b.json                  Groq free-tier baseline
└── .gitignore
```

**Earlier negative result (still relevant context):**
The previous HART pipeline tried to *blend* the vector cosine score and a
header-alignment score with a single α weight. On HiTab dev it never beat
plain markdown serialization on R@1 / nDCG / MRR — that negative finding
motivated this work, which keeps the two stores fully separate and routes
queries through different stages instead of blending.

---

## Quickstart

Hardware tested on: RTX 3060 Ti (8 GB VRAM), WSL2 Ubuntu, Python 3.12.

Data is not vendored. You need:

- HiTab dev split (`microsoft/HiTab`) extracted somewhere on disk.
- A Chroma collection containing one vector per table (the package will
  re-use an existing collection named `plain_markdown_bge_large_en_v1_5`;
  build your own with any `sentence-transformers` model and serializer).

```bash
# install
pip install -e rag-agent/

# local Qwen-7B (no API key needed; needs ~5 GB VRAM)
python rag-agent/scripts/run_eval.py \
    --llm local:Qwen/Qwen2.5-7B-Instruct \
    --data-dir   /path/to/HiTab/data/hitab \
    --chroma-dir /path/to/chroma_db \
    --per-class 8 --limit 40 \
    --retriever-device cpu \
    --out rag-agent/results/local_qwen7b_v3.json

# Groq free-tier
GROQ_API_KEY=...  python rag-agent/scripts/run_eval.py \
    --llm groq:llama-3.1-8b-instant \
    --data-dir   /path/to/HiTab/data/hitab \
    --chroma-dir /path/to/chroma_db \
    --per-class 8 --limit 40 \
    --out rag-agent/results/groq_llama3.1_8b.json

# strongest config — Qwen reads, Groq-70B extracts cells
GROQ_API_KEY=...  python rag-agent/scripts/run_eval.py \
    --llm local:Qwen/Qwen2.5-7B-Instruct \
    --symbolic-llm groq:llama-3.3-70b-versatile \
    --data-dir   /path/to/HiTab/data/hitab \
    --chroma-dir /path/to/chroma_db \
    --per-class 8 --limit 40 --retriever-device cpu \
    --out rag-agent/results/mixed.json
```

Each result JSON contains the full per-query trace (intent, plan
stages run, vector top-5, verified top-5, symbolic plan, resolved
cells, AST value, reader raw output, EM/NM verdict). The headline
numbers above can be re-derived from those traces.

---

## Documentation

- [`rag-agent/README.md`](rag-agent/README.md) — package overview
- [`rag-agent/EXPERIMENTS.md`](rag-agent/EXPERIMENTS.md) — full experiment
  report: hardware, software, every metric with paper citation,
  v1 → v3.1 run-by-run progression, the four audit-bug fixes, and the
  multi-seed / ablation / 70B-extractor / CI tables.

---

## License

[MIT](https://spdx.org/licenses/MIT.html)
