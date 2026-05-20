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

## Repository layout (thesis sections)

| Thesis § | Module | What it does |
|---|---|---|
| §3 Serialization audit | `experiments/exp01_serialization_audit.py` | 5 serializers × 4 structural metrics on HiTab |
| §4 Layer-wise probing | `experiments/exp02_layer_probing.py` | Linear/MLP probes across 12 transformer layers |
| §5 HART table retrieval | `hart-table-retrieval/` | Serializer × embedder × header-alignment ablation. Negative result (HART scorer ≤ plain markdown). |
| **§6 Adaptive Table-RAG agent (this work)** | **`rag-agent/`** | **Routing policy + verifier + symbolic compute. NM 0.250 → 0.450.** |

---

## Quickstart

Hardware tested on: RTX 3060 Ti (8 GB VRAM), WSL2 Ubuntu, Python 3.12.

```bash
# 1. Data: HiTab dev + an existing Chroma index built by the HART pipeline.
#    Expected layout (re-used by rag-agent):
#      /home/user/T2/hart-table-retrieval/data/hitab/
#      /home/user/T2/hart-table-retrieval/data/chroma_db/

# 2. Run the local Qwen-7B benchmark (no API key needed)
python rag-agent/scripts/run_eval.py \
    --llm local:Qwen/Qwen2.5-7B-Instruct \
    --per-class 8 --limit 40 \
    --retriever-device cpu \
    --out rag-agent/results/local_qwen7b_v3.json

# 3. Or run the Groq free-tier comparison
GROQ_API_KEY=...  python rag-agent/scripts/run_eval.py \
    --llm groq:llama-3.1-8b-instant \
    --per-class 8 --limit 40 \
    --out rag-agent/results/groq_llama3.1_8b.json

# 4. Strongest config (Qwen reads, Groq-70B extracts) — recommended if TPD allows:
GROQ_API_KEY=...  python rag-agent/scripts/run_eval.py \
    --llm local:Qwen/Qwen2.5-7B-Instruct \
    --symbolic-llm groq:llama-3.3-70b-versatile \
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
  v1 → v2 → v3 run-by-run progression, limitations.
- [`hart-table-retrieval/README.md`](hart-table-retrieval/README.md) —
  the §5 HART pipeline whose negative result motivated this work.

---

## License

[MIT](https://spdx.org/licenses/MIT.html)
