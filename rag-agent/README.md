# rag-agent — retrieval completeness for hierarchical-table QA

Research code for a masters thesis on RAG over **hierarchical tables** (HiTab,
MultiHiertt, RealHiTBench, FinQA, WikiSQL, AIT-QA, IM-TQA).

**Rule this repo runs on** (from `NEXT.md`): *every number carries the results
file it came from.* A number without a source does not go in a document. This
README follows it — each figure below names its file under `results/`.

---

## The problem

An aggregation question ("total revenue across all divisions") needs **every**
operand cell. Miss one and the sum is wrong — there is no partial credit at the
answer. But the retrieval literature scores per-fact recall, which cannot see
this failure: MultiHiertt reports 76.4% recall@10 while its own error analysis
lists **Missing Operand at 21%** (arXiv 2206.01347, Table 6).

So the metric this repo optimises is **Operand-Set Completeness (OSC)**: 1 iff
every gold operand was retrieved, 0 otherwise. `rag_agent/eval/operand_set.py`.

## Honest positioning

Read this before quoting anything as novel.

| Component | Status |
|---|---|
| Cell → (caption + row path + col path) sentence as the retrieval unit | **Prior art.** MultiHiertt (2206.01347, 2022) does exactly this, verbatim. Adopt and cite it; do not present it as proposed. |
| Header-path vs leaf-only serialization gain | **Prior art.** OHD (2602.01969) Table 2 ablation: 53.33 (markdown) → 60.07 (lineage), +6.7pp. |
| Set-level all-or-nothing completeness as a *retrieval objective* | Open — the literature measures per-fact recall. |
| Language-independent structural aggregate-row detection + injection | Open; HiTab-scoped case study, not a general method. |

Also relevant and missing from the related-work notes: IEEE Access 2025,
DOI 10.1109/ACCESS.2025.3569872 (header hierarchy for tabular RAG, table-level
granularity). Patent-side, US11836445B2 (Microsoft, active) covers header/data
area detection with merged-cell spans.

## What holds up

**Serialization vs baseline, RealHiTBench** (raw hierarchical tables, strict
`em_norm`, McNemar) — same retriever, same k, only the serialization differs:

| solver | n | S1 flat | S2 header-path | Δ | p |
|---|---|---|---|---|---|
| gpt-5.1 | 94 | .160 | .309 | +.149 | .0043 |
| gpt-4.1-mini | 94 | .170 | .277 | +.106 | .0213 |
| llama-3.3-70b | 44 | .205 | .341 | +.136 | .0703 |

→ `results/realhitbench_s1_vs_s2*`. Consistent across three solver families, so
the effect is not a property of one reader. Do not compare these against the
RealHiTBench leaderboard — that uses an LLM judge, this uses a local scorer.

**Semantic header-path resolver, HiTab** — `results/resolver_osc_matched_dev.json`.
The lexical scorer in `resolve_against_table` keeps only header paths scoring
above zero, discarding every header sharing no token with the question: 28.0% of
gold rows, 31.3% of gold columns. Nothing is structurally unreachable — every
gold row/column has a real non-empty header path. Wiring the existing
`EmbedResolver` into the operand-targeted path (`embed_resolver=True`) lifts
both-axes resolution .285 → .537 and OSC:

| dev, m≥2 | lexical | semantic | budget-matched lexical | Δ vs matched | p |
|---|---|---|---|---|---|
| k=10 | .4348 | .5466 | .4720 | +.075 | .111 |
| k=20 | .5963 | .6646 | .5714 | +.093 | .036 |

`train` split run: `results/resolver_osc_matched_train.json`.

**Structural total-row injection, HiTab** — a case study, not the headline.
Index-time detection, zero per-query cost, official `hitab_exact_match`
.395 → .500 on gpt-oss-120b (n=86, p=.022) → `results/h6_rerun_20260707/`.
Applies to ~37% of HiTab queries and does not transfer: WikiSQL has no total
cells, FinQA tables are too small (median 5 rows).

## What failed (kept, because the mechanisms are clean)

**The completeness gate fires zero times** at base k≥3.
`results/pipeline_osc_asdescribed.json`, `results/gate_k{1,2,3}.json`:

| base k | fired | OSC before → after | Δ |
|---|---|---|---|
| 1 | 12/214 | .2103 → .2290 | +.019 |
| 2 | 3/214 | .2710 → .2757 | +.005 |
| 3 | 0/214 | .3411 | 0 |
| 5 | 0/214 | .4252 | 0 |

All 12 recoveries came from the cheapest rung (a wider k); the structural and
enumeration rungs never fired at any k. The gate is gold-free — it asks whether
the operands *the decomposer requested* were retrieved — and the retriever
already searches per operand, so the question is near-tautological. It bounds
retrieval failures; the binding constraint here is decomposition accuracy.

**Scope enumeration is a budget change in disguise.** Unioning the resolved
header scope into the top-k lifted OSC .4252 → .5654, but the evidence set grew
from 10 to 19 cells; against a per-query budget-matched control the gain fell to
+.014 (p=.55). Dropped.

> **Standing rule from that failure:** any arm that grows the evidence set must
> report a per-query budget-matched control — the same cell count filled by
> similarity rank alone — before its delta is quoted.

## Layout

```
rag_agent/
  serialize/          S1 flat, S2 header-path, S3 caption-sentence index units
  stores/             header-grid reconstruction (rowspan/colspan, carry-fill)
  query/              header-path intent resolution (lexical + EmbedResolver)
  retrieve/           hybrid index, operand-targeted retrieval, structural
                      attribution, completeness gate, header enumeration
  eval/               OSC, per-cell recall, official hitab_exact_match
  generate/           codegen / direct answerers
scripts/              one experiment leg per file, each writing results/<name>.json
results/              committed measurements — the source of every number above
tests/                pytest, no network, fake embedders
```

## Running

```bash
.venv/bin/python -m pytest tests/ -q

PYTHONPATH=. .venv/bin/python scripts/resolver_osc_matched.py --split dev
PYTHONPATH=. .venv/bin/python scripts/pipeline_osc_asdescribed.py --split dev --k 5

GROQ_API_KEY=... PYTHONPATH=. .venv/bin/python scripts/answer_accuracy_resolver.py \
    --solver-model openai/gpt-oss-120b --codegen-max-tokens 1024 --resume
```

LLM legs append one record per query and resume with `--resume`, so a daily
token cutoff never loses work.

## Open

- Answer-accuracy leg for the resolver fix (`scripts/answer_accuracy_resolver.py`)
  — retrieval gains do not convert automatically: total-row injection converted
  on gpt-oss-120b and not at all on llama-3.1-8b.
- Resolver fix on MultiHiertt / RealHiTBench.
- Row-axis reconstruction sits at .582 on real grids; 41% of tables encode the
  hierarchy in a single stub column, so the information is absent, not misread.
