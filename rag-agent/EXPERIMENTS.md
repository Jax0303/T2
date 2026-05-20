# rag-agent — Experiment Report

End-to-end table-QA evaluation of the `rag-agent` package on HiTab dev,
comparing two **free** LLM backends and isolating where the pipeline
helps, where it hurts, and where the genuine model-capability ceiling is.

This document covers: setup, methodology, every benchmark run made
(v1 / v2 / v3 — with the bug fixes that produced each), and the honest
findings. It is meant to be self-contained for a thesis appendix.

---

## 1. Goal

Build a non-HART RAG agent for HiTab where:

- **Original 2-D table data** and the **vector-DB store** are kept
  *separate*. The agent compares the two at verification time rather
  than re-reading the serialized chunk that the retriever already saw.
- Queries are routed by **rule-based intent classification** that
  mirrors the HiTab paper's appendix categories (aggregation array +
  Excel formula op-count). Stages can be skipped when not needed
  (e.g. reasoning-only questions bypass retrieval).
- Arithmetic/formula questions are answered by **symbolic compute**:
  the LLM emits a `{cells, expression}` JSON; we resolve the cells via
  header-path lookup in the original store and evaluate the expression
  with an AST-whitelist sandbox. The LLM never executes the arithmetic.
- Use **only free LLM APIs / local models** (no OpenAI, no paid keys).
- Report metrics that come **verbatim from the HiTab and HART papers**
  (Recall@k / MRR / nDCG / Exact-Match / Numeric-Match), plus one
  custom metric (symbolic-exec accuracy) explicitly inspired by HiTab
  Table 9's formula-supervised execution-accuracy.

---

## 2. Setup

### Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 3060 Ti, 8 GB VRAM |
| CPU | x86-64, used for the embedder when LLM holds GPU |
| RAM | 10 GB WSL2-allocated, `vmIdleTimeout=-1` |

For the home-PC runs (3060 Ti / 8 GB), the embedder is on CPU and the
LLM (Qwen-7B 4-bit, ~5 GB) holds the GPU. On a ≥16 GB card both can
share CUDA — pass `--retriever-device cuda`.

### Software

- WSL2 Ubuntu, Python 3.12
- PyTorch 2.12.0 + CUDA 12.6
- transformers 5.8.1, bitsandbytes 0.49.2 (NF4 4-bit)
- sentence-transformers 5.5.0 (`BAAI/bge-large-en-v1.5`, 1024-dim)
- chromadb 1.5.9 (persistent dir reused from the HART pipeline)
- groq 1.2.0 (free-tier API; key required via `GROQ_API_KEY`)

### Dataset

HiTab dev split, 1,671 query-table pairs over **540 unique tables**.
We use the per-question supervision (`aggregation`, `answer_formulas`)
only for difficulty stratification and gold-answer scoring — the agent
itself never sees these fields.

Data layout (re-using the existing HART data dir):

```
/home/user/T2/hart-table-retrieval/
├── data/hitab/data/
│   ├── dev_samples.jsonl
│   └── tables/{hmt,raw}/*.json
└── data/chroma_db/                    bge-large + multilingual-e5 indexes
```

The Chroma collection used here is `plain_markdown_bge_large_en_v1_5`
(540 docs, one per table). It was built by the HART indexing script
and is reused as-is — no re-embedding for this experiment.

### Query stratification

We use the same difficulty taxonomy as `hart-table-retrieval/scripts/
run_hard_query_eval.py`, derived directly from HiTab's gold supervision:

| Class | Population in dev | What it tests |
|---|---:|---|
| `multi_op_formula` | 37 | Excel formulas with ≥2 ops, e.g. `=(B+C+D)/E` |
| `arithmetic_agg` | 139 | sum / diff / div / avg / range |
| `pair_or_topk_arg` | 153 | "X or Y?" — pair-argmax/min, top-k pick |
| `single_arg` | 93 | argmax / argmin / max / min |
| `comparison_or_count` | 54 | greater_than / less_than / opposite / counta |

40 queries are sampled with `per_class=8`, `seed=0`. We deliberately
match the existing hard-query eval's setup so numbers are directly
comparable to the older Sidecar+CoT result (overall NM = 0.250).

---

## 3. Pipeline

Each query passes through these stages; **stages are skipped based on
the classifier's intent**, not silently dropped — the trace records what
ran and what was skipped.

```
                    ┌──────────────────────────────────────┐
       query  ────▶ │ (1) classify_query                   │
                    │   rule-based regex/keyword,          │
                    │   outputs one of 6 intents           │
                    └──────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ (2) plan_stages                      │
                    │   reasoning_only → skip retrieve+ver │
                    │   arith/multi_op → +symbolic         │
                    └──────────────┬───────────────────────┘
                                   ▼
       ┌────────────────────┐    ┌──────────────────────────────────┐
       │ VectorStore        │◀── │ (3) retrieve                     │
       │ Chroma + bge-large │    │   top-20 vectors → top-5 tables  │
       └────────────────────┘    └──────────────┬───────────────────┘
                                                ▼
                         ┌──────────────────────────────────────┐
                         │ (4) verify                           │
                         │   keyword overlap (query ↔ headers)  │
                         │   numeric overlap (query ↔ cells)    │
                         │   composite = 0.7·vec + 0.3·verify   │
                         │   rerank top-5                       │
                         └──────────────┬───────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │ (5a) SYMBOLIC (arith / multi_op)     │
                         │   LLM emits JSON cells+expression    │
                         │   header-path resolve → AST eval     │
                         │   gated: adopt only if non-trivial   │
                         └──────────────┬───────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │ (5b) LLM reader (fallback / lookup)  │
                         │   table+query → "Final answer: …"    │
                         └──────────────────────────────────────┘
```

### 3.1 Query classifier (`router/query_classifier.py`)

Pure regex/keyword. Six output labels matching the HiTab supervision
taxonomy. Selected design rules learned during the audit:

- Explicit math symbols `[+\-*/]` appearing ≥2 times → `multi_op_formula`.
- `_ARITH_TRIGGERS` with ≥2 distinct hits (or `_MULTI_PAT`) → `multi_op_formula`.
- Entity-cue questions ("who/which X had higher/lower Y?") → `single_arg`,
  **not** `arithmetic_agg`, even when "higher proportion" or similar appears.
  This single rule fixed a pair-class regression discovered in audit (§5).
- `total` is only an aggregation cue when **not** followed by "row / column /
  of the …" — otherwise it is a row label.
- `_REASONING_PAT` (definition-like opener with no table cues) →
  `reasoning_only` → skip retrieve / verify / symbolic entirely.

### 3.2 Original store (`stores/original_store.py`)

Wraps HiTab's parsed JSON. Exposes:

- `data` — 2-D list of cell values (one matrix per table).
- `top_paths[col]` / `left_paths[row]` — header path lists (drop the
  synthetic `<TOP>` / `<LEFT>` / `<ROOT>` sentinels).
- `find_cols_by_header(token)` / `find_rows_by_header(token)` — robust
  header matching: split the LLM's path on any common separator
  (`'>', '::', '/', '|'`) and require every token to be a substring of
  the joined actual path. Case-insensitive. **This is the resolver
  the symbolic-compute path depends on.** A bug in this matcher was
  the cause of the spurious 0.000 multi_op_formula score in v1 (§5).
- `resolve(row_header, col_header)` — finds the most-specific matching
  (row, col) pair and returns the cell value.
- `excel_ref_to_rc("B21")` — kept for ad-hoc debugging only.

### 3.3 Vector store (`stores/vector_store.py`)

Thin wrapper around the existing Chroma collection. The embedder is
`BAAI/bge-large-en-v1.5` (1024-dim, GPU when available). One vector
per table (after the HART indexing pipeline), so `top_k_tables = 5`
returns 5 distinct candidates after dedup-by-table.

### 3.4 Verifier (`retrieve/verifier.py`)

The "원본과 벡터 DB 동시 비교/검증" step. Given the vector top-K, look
each candidate up in the **original store** (not the chunk that was
retrieved) and score:

- `keyword_overlap` — Jaccard between query keywords and the union of
  the candidate's title + every header path token.
- `numeric_overlap` — fraction of query numbers (parsed by regex) that
  appear as an exact match in any cell. If the query has no numbers,
  this is set to 1.0 (neutral) and only the keyword signal matters.
- `confidence = 0.6 · keyword + 0.4 · number` (or just keyword when no
  numbers in the query).

Final rerank: `final_score = 0.7 · vector_score + 0.3 · verify_confidence`.

### 3.5 Symbolic compute (`extract/`)

For `arithmetic_agg` / `multi_op_formula`:

1. The LLM is shown a compact rendering of the top-1 verified table
   (full top-header paths + left-header paths + first 30 rows of data)
   and asked to output JSON in this exact schema:

   ```json
   {
     "cells": [
       {"var": "x1", "row_header": "...", "col_header": "..."},
       {"var": "x2", "row_header": "...", "col_header": "..."}
     ],
     "expression": "x1 - x2"
   }
   ```

2. We parse the JSON, resolve each cell via `OriginalStore.resolve()`,
   and evaluate `expression` with an AST whitelist (`+ - * / ( )`,
   constants, `Name` lookups only — no `Call`, no `Attribute`, no
   `__import__`). The evaluator is unit-tested to reject
   `__import__("os").system(...)`.

3. **Adoption gate** (added after audit): only replace the reader's
   answer with the symbolic answer when

   ```
   op_count(expression) ≥ 2
        OR
   (intent == arithmetic_agg AND op_count ≥ 1 AND cells ≥ 2)
   ```

   Otherwise the symbolic attempt is recorded in the trace but the
   reader speaks. Without this gate, a spurious `x1 - x2` extraction
   on a pair-question can displace a correct name-answer the reader
   would have produced (§5, bug #3).

### 3.6 LLM backends (`llm/`)

| Spec | Notes |
|---|---|
| `groq:llama-3.3-70b-versatile` | **Strongest free model.** 100k TPD on free tier — too low for a full 48-query run with ~2.4k tokens/query of table context. Partial 41-query run only. |
| `groq:llama-3.1-8b-instant` | 500k TPD on free tier — fits a full 40-query run. **Weak at JSON-format following.** |
| `local:Qwen/Qwen2.5-7B-Instruct` | 4-bit NF4 on 3060 Ti (~5 GB VRAM). Greedy decoding. **Strongest among the three for table-reader tasks.** |

The `BaseLLM` interface is `complete(system, user, max_tokens) -> str`,
so the reader and the cell-extractor can be different models. We did
not use `gpt-4o-mini` or any paid model.

---

## 4. Metrics

All retrieval metrics are computed on the **verified top-5** (after
rerank). Answer metrics are computed on whichever of {symbolic, reader}
the gate selected.

| Metric | Definition | Source |
|---|---|---|
| Recall@1, Recall@5 | gold table in top-k after rerank | HiTab, HART |
| MRR | mean reciprocal rank of gold table | HiTab, HART |
| nDCG@10 | binary single-gold relevance, log2 discount | HART |
| Exact Match (EM) | lowercase-stripped string equality, any element of gold list | HiTab §5 |
| Numeric Match (NM) | rel-tol ±2% with HiTab variants: ×100 (%), ÷100 (fraction), abs() (opposite/sign). Falls back to case-insensitive substring for string gold | HiTab §5 + matches the existing hard-query bench |
| Symbolic exec accuracy | for `arithmetic_agg`/`multi_op_formula`: did the AST eval over extracted cells produce a number that matches gold under NM? | Custom, inspired by HiTab Table 9 formula-supervised exec-acc |

Per-class breakdown matches `HARD_CLASSES`: same 5 difficulty bins as
the existing eval (`single_op_formula` and `simple_lookup` are not
sampled in this 40-query subset, so they appear with `n=0`).

---

## 5. Runs and bug-fix progression

Three benchmark runs were performed on the same 40-query stratified
sample (seed=0). Each was triggered by a discovered defect; reporting
all three is required to honestly explain how the final numbers were
obtained.

### v1 — initial run (Qwen-7B 4-bit)

```
class                      n   R@1_v   R@1   R@5   MRR   nDCG    EM    NM   sym_atm sym_corr
multi_op_formula           8   0.625  0.500 0.875 0.688 0.737  0.000 0.000  0.000   0.000
arithmetic_agg             8   0.375  0.375 0.750 0.504 0.565  0.125 0.125  0.125   0.000
pair_or_topk_arg           8   0.500  0.875 1.000 0.938 0.954  0.750 0.875  0.000   0.000
single_arg                 8   0.625  0.750 0.750 0.750 0.750  0.500 0.500  0.000   0.000
comparison_or_count        8   0.750  0.875 1.000 0.917 0.938  0.125 0.750  0.000   0.000
OVERALL                   40   0.575  0.675 0.875 0.759 0.789  0.300 0.450  0.025   0.000
```

**v1 finding**: `sym_attempted = 0.025` (1/40). Symbolic almost never
fired. Multi_op_formula NM = 0.000 looked credible at first ("readers
can't do multi-cell arithmetic, this is the existing literature").

### Bug #1 (audit discovery)

The cell-extractor prompt renders each column header as
`"col[i]: A > B > C"` using `' > '` as the separator, but
`OriginalStore.find_cols_by_header()` was joining the actual path with
`' :: '` and doing a single substring check on the whole LLM string:

```python
# what the LLM emits:  "size of company > medium companies > 100-249"
# what resolve checks:   "size of company :: medium companies :: 100-249"
# substring of one in the other → False → unresolved_cell
```

Every multi_op_formula symbolic attempt failed with
`error="unresolved_cell:x1"`. The 0.000 was a **measurement artifact,
not a real model limitation**. Fix: split the LLM's path on any common
separator and require every token to be a substring of the joined
actual path.

### v2 — after header-resolver fix

```
class                      n   R@1_v   R@1   R@5   MRR   nDCG    EM    NM   sym_atm sym_corr
multi_op_formula           8   0.625  0.500 0.875 0.688 0.737  0.000 0.000  0.500   0.000
arithmetic_agg             8   0.375  0.375 0.750 0.504 0.565  0.125 0.125  0.625   0.125
pair_or_topk_arg           8   0.500  0.875 1.000 0.938 0.954  0.625 0.750  0.125   0.000
single_arg                 8   0.625  0.750 0.750 0.750 0.750  0.500 0.500  0.000   0.000
comparison_or_count        8   0.750  0.875 1.000 0.917 0.938  0.125 0.750  0.625   0.375
OVERALL                   40   0.575  0.675 0.875 0.759 0.789  0.275 0.425  0.375   0.100
```

**v2 finding 1**: Symbolic-attempted shot up from 0.025 → 0.375. Three
arithmetic classes now produce numeric symbolic answers (sym_correct =
0.125 in arithmetic_agg, **0.375 in comparison_or_count**). The pipeline
mechanically works.

**v2 finding 2**: pair_or_topk_arg **regressed from 0.875 → 0.750**.
Inspection of the row:

```
query = "who had a statistically higher proportion of being missed in 2011?"
gold  = ["asia or oceania"]
classified_as = arithmetic_agg          # "proportion" in _ARITH_PAT
symbolic fired → x1 - x2 = 8.4
reader (would have said "asia or oceania") was overridden  → NM=0
```

Two bugs revealed:

- **Bug #2 — classifier misroute**: "who/which X had higher/lower Y?"
  is asking for an entity name, not a computed value, even when a
  noun like "proportion" appears. Fix: add `_ENTITY_QUESTION_PAT`
  (`^(who|which|what|where) …`) and route those to `single_arg`
  whenever an arg/pair cue is also present. Extend `_ARG_PAT` to
  include `higher / lower / more / fewer / greater`.
- **Bug #3 — symbolic over-firing**: a 1-operator extraction
  (`x1 - x2`) was being adopted unconditionally. Fix: add the
  adoption gate described in §3.5 — adopt symbolic only on ≥2 ops
  or on `intent == arithmetic_agg` with ≥1 op and ≥2 cells.

**v2 finding 3 (real)**: `multi_op_formula` is genuinely 0.000 *after*
the header-resolver fix. 3-of-8 plans now resolve to numbers, but every
one is wrong — Qwen-7B picks the wrong cells / inverts denominators /
shifts a row. Example: gold `=C14+C20+C22 = 51.5`, LLM extracts a
3-cell sum `(x1+x2+x3) = 49.5`. The arithmetic itself is correct; the
**cell selection** is the bottleneck.

### v3 — after classifier + symbolic-gate fixes (FINAL)

```
class                      n   R@1_v   R@1   R@5   MRR   nDCG    EM    NM   sym_atm sym_corr
multi_op_formula           8   0.625  0.500 0.875 0.688 0.737  0.000 0.000  0.375   0.000
arithmetic_agg             8   0.375  0.375 0.750 0.504 0.565  0.125 0.125  0.500   0.125
pair_or_topk_arg           8   0.500  0.875 1.000 0.938 0.954  0.750 0.875  0.000   0.000
single_arg                 8   0.625  0.750 0.750 0.750 0.750  0.500 0.500  0.000   0.000
comparison_or_count        8   0.750  0.875 1.000 0.917 0.938  0.125 0.750  0.625   0.375
OVERALL                   40   0.575  0.675 0.875 0.759 0.789  0.300 0.450  0.300   0.100
```

- pair_or_topk_arg **healed** to 0.875 (the classifier no longer routes
  "who … higher proportion …" to arithmetic).
- Symbolic precision **up**: `sym_attempted` 0.375 → 0.300 (gate filters
  out trivial single-op extractions on non-arith questions), `sym_correct`
  unchanged at 0.100 — same wins, fewer false fires.
- Overall NM restored to **0.450** (matched v1's headline but without
  the broken `pair = arith` luck).

### Groq comparison (`llama-3.1-8b-instant`, 40 queries, identical seed)

```
class                      n   R@1_v   R@1   R@5   MRR   nDCG    EM    NM   sym_atm sym_corr
multi_op_formula           8   0.625  0.500 0.875 0.688 0.737  0.000 0.000  0.000   0.000
arithmetic_agg             8   0.375  0.375 0.750 0.504 0.565  0.000 0.000  0.000   0.000
pair_or_topk_arg           8   0.500  0.875 1.000 0.938 0.954  0.000 0.125  0.000   0.000
single_arg                 8   0.625  0.750 0.750 0.750 0.750  0.000 0.000  0.000   0.000
comparison_or_count        8   0.750  0.875 1.000 0.917 0.938  0.000 0.625  0.125   0.000
OVERALL                   40   0.575  0.675 0.875 0.759 0.789  0.000 0.150  0.025   0.000
```

- Retrieval is identical (same Chroma + bge-large, same verifier).
- Reader is **3× worse** on NM (0.150 vs 0.450). Llama-3.1-8B rarely
  produces a short `Final answer:` line and almost never emits a
  parseable JSON for cell extraction (sym_attempted = 0.025).

### Groq Llama-3.3-70B (partial, 41 queries)

Hit the free-tier TPD limit (100k tokens/day) at query 41/48 because
each table context costs ~2k tokens at this model size. The partial run
showed **4 symbolic successes** (vs Qwen-7B's 4) and a per-class NM
profile similar to Qwen-7B's, suggesting that under TPD-unlimited
conditions the 70B would be the most reliable cell-extractor. We do
not include partial numbers in the headline table because the last
class (`single_op_formula` / `simple_lookup`) was cut off.

---

## 6. Headline summary (v3, final)

| Metric | Value | Notes |
|---|---:|---|
| R@1 (vector only) | 0.575 | bge-large-en-v1.5 on `plain_markdown` |
| **R@1 (after verifier rerank)** | **0.675** | +10 pp from the original-store cross-check |
| R@5 (after rerank) | 0.875 | |
| MRR | 0.759 | |
| nDCG@10 | 0.789 | |
| EM | 0.300 | |
| **Numeric Match** | **0.450** | vs existing hard-query bench (Sidecar+CoT): 0.250 |
| Symbolic exec acc (overall) | 0.100 | 4 of 40 queries resolved entirely by the deterministic path |
| Symbolic exec acc (`comparison_or_count`) | 0.375 | best class |

Reader = local Qwen-2.5-7B-Instruct, 4-bit NF4.

---

## 7. Findings

1. **Verifier rerank is robust**. Cross-checking vector candidates
   against the original 2-D store (keyword + numeric overlap on
   headers/cells) lifts R@1 by 10 pp on this hard subset. This is the
   most consistent positive result.
2. **Free-LLM choice matters more than the rerank**. Same retrieval,
   same verifier, same symbolic path: Qwen-2.5-7B-4bit ⇒ NM 0.450 vs
   Llama-3.1-8B-Instant ⇒ NM 0.150. Qwen's table-QA tuning shows.
3. **Symbolic compute fires reliably on comparison/count classes**
   (3/8 fully deterministic answers on `comparison_or_count`). For
   `arithmetic_agg` it fires but the LLM's cell selection is only ~1/8
   correct.
4. **`multi_op_formula = 0.000` is real, not a bug**, *after* the
   audit. Cell extraction succeeds (3/8 produce numbers via AST eval)
   but the LLM picks the wrong cells / wrong formula structure every
   time. This is Qwen-7B's genuine ceiling.
5. **The audit itself yielded three concrete bugs** (header separator
   mismatch, classifier misroute on "who had higher proportion",
   symbolic over-firing without an op-count gate). All are fixed in
   the v3 numbers above. None of the bugs would have been caught by
   tests alone — they were found by reading per-query traces.

---

## 8. Limitations and threats to validity

- **n = 8 per class.** Standard error per cell is ~17 pp. The 0/1 endpoints
  are common at this sample size. The headline numbers should be read
  as point estimates, not population means.
- **No TabFact / FeTaQA evaluation.** Our claim is specific to HiTab
  hard queries. The Sidecar paper already showed that on uniformly-
  structured Wikipedia tables (TabFact) the verifier rerank *hurts*
  by ~1.5 pp because there are no discriminative header keywords.
- **`reasoning_only` intent is never observed in HiTab.** That branch of
  the policy is correct by construction but unexercised here.
- **Symbolic path uses Qwen-7B for cell extraction.** The 70B partial
  run hints that a stronger extractor would meaningfully lift
  `multi_op_formula`. A follow-up should pin the reader to Qwen-7B and
  swap only the extractor (`--symbolic-llm`) to isolate that effect.
- **Greedy decoding only.** No temperature/top-p sweep.

---

## 9. Reproducing

```bash
# 0. expects /home/user/T2/hart-table-retrieval/{data/hitab, data/chroma_db} present
# 1. local Qwen-7B run (no API key needed):
/home/user/T2/hart-table-retrieval/.venv/bin/python rag-agent/scripts/run_eval.py \
    --llm local:Qwen/Qwen2.5-7B-Instruct \
    --per-class 8 --limit 40 \
    --retriever-device cpu \
    --out rag-agent/results/local_qwen7b_v3.json

# 2. Groq Llama-3.1-8B run (free tier):
GROQ_API_KEY=... /home/user/T2/hart-table-retrieval/.venv/bin/python \
    rag-agent/scripts/run_eval.py \
    --llm groq:llama-3.1-8b-instant \
    --per-class 8 --limit 40 \
    --out rag-agent/results/groq_llama3.1_8b.json

# 3. mixed: Qwen as reader, 70B Groq as cell-extractor (recommended if TPD allows):
GROQ_API_KEY=... /home/user/T2/hart-table-retrieval/.venv/bin/python \
    rag-agent/scripts/run_eval.py \
    --llm local:Qwen/Qwen2.5-7B-Instruct \
    --symbolic-llm groq:llama-3.3-70b-versatile \
    --per-class 8 --limit 40 \
    --retriever-device cpu \
    --out rag-agent/results/mixed.json
```

All output JSONs include per-query traces (vector top-5, verified
top-5, intent, plan stages run, symbolic plan + resolved cells + AST
value, reader raw output, answer + EM/NM verdict) so the headline
numbers can be re-derived offline.

---

## 10. Files added in this experiment

```
rag-agent/
├── README.md                        package overview + quickstart
├── EXPERIMENTS.md                   this file
├── rag_agent/
│   ├── stores/                      OriginalStore (parsed 2-D) + VectorStore (Chroma + GPU)
│   ├── router/                      query_classifier + policy
│   ├── retrieve/                    verifier (keyword/number overlap) + rerank
│   ├── extract/                     cell_extractor (JSON) + symbolic_eval (AST whitelist)
│   ├── llm/                         BaseLLM, GroqLLM, LocalQwenLLM, build_llm()
│   ├── agent.py                     orchestrator (adoption gate, trace)
│   └── eval/metrics.py              R@k, MRR, nDCG, EM, NM, difficulty_class
├── scripts/
│   ├── run_eval.py                  benchmark entry point
│   └── smoke_test.py                offline integration test (no LLM)
└── results/
    ├── groq_llama3.1_8b.json        Groq 8B, 40 queries
    ├── groq_llama3.3_70b.json       Groq 70B, partial (TPD limit at 41/48)
    ├── local_qwen7b.json            v1 — pre-fix
    ├── local_qwen7b_v2.json         v2 — header-resolver fix
    └── local_qwen7b_v3.json         v3 — classifier + symbolic-gate fix (FINAL)
```
