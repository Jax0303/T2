# rag-agent — retrieval completeness for hierarchical-table QA

Research code for a masters thesis on RAG over **hierarchical tables** (HiTab,
MultiHiertt, RealHiTBench, FinQA, WikiSQL, AIT-QA, IM-TQA).

Start at the repo-root [`README.md`](../README.md) for how the contribution is
stated and what is prior art; [`STATUS-2026-08-06.md`](STATUS-2026-08-06.md) is the
current handoff note.

**Rule this repo runs on** (from `NEXT.md`): *every number carries the results
file it came from.* A number without a source does not go in a document. This
README follows it — each figure below names its file under `results/`.

---

## The problem

An aggregation question ("total revenue across all divisions") needs **every**
operand cell. Miss one and the sum is wrong — there is no partial credit at the
answer. But the retrieval literature scores per-fact recall, which cannot see
this failure — a healthy recall@10 is compatible with most aggregation queries
missing at least one operand.

⚠️ Do **not** motivate this with MultiHiertt's error table. Its Table 6
(Wrong Operand or Span 43%, Missing Operand 21%) classifies the *generated
program*, and the paper never splits those into "retriever did not return the
cell" vs "reasoner did not use it". See the root README.

So the metric this repo optimises is **Operand-Set Completeness (OSC)**: 1 iff
every gold operand was retrieved, 0 otherwise. `rag_agent/eval/operand_set.py`.

## Honest positioning

Read this before quoting anything as novel.

| Component | Status |
|---|---|
| Cell → (caption + row path + col path) sentence as the retrieval unit | **Prior art.** MultiHiertt/MT2Net (2206.01347, 2022) §4 does exactly this, verbatim — including extracting the hierarchy by script rather than annotation, and the flat-vs-hierarchical comparison. Adopt and cite it; do not present it as proposed. |
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
Index-time detection, zero per-query cost.

⚠️ **The population-lift numbers this README used to quote (.395 → .500, n=86,
p=.022) are withdrawn** — they were an artifact of where a token-limited run
stopped, with `--flips-first` concentrating the 10 flipped queries at the front.
Scored to completion the effect is +.019, p=.581 (gpt-4o, n=161). No truncated-
sample p-value from this leg is citable. See `RESEARCH_STRUCTURE.md` §4.3.

What holds: injection changes retrieval on 10/161 queries (6.2%), and on exactly
those, accuracy goes .00 → .90 (gpt-oss-120b, 9:0) and .00 → .50 (gpt-4o, 5:0),
zero losses. Ceiling on population lift ≈ 5.6pp. Also HiTab-only: WikiSQL has no
total cells, FinQA tables are too small (median 5 rows).

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

## Diagnoses OSC makes visible

Not repeated here — tables and p-values in the root [`README.md`](../README.md):
reranking breaks the set only under a narrow budget (@50 the effect is gone;
`results/operand_collision_rerank_n300.json` and the pool-size sweep
`results/ea_pool_size_sweep_*.json` agree independently), and the header-path gain
is lexical with **no detectable order effect** over six shuffle draws
(`results/shuf_spread_s{1..5}.json`).

## Open

Priority order (see [`STATUS-2026-08-06.md`](STATUS-2026-08-06.md)):

1. **Query decomposition.** `osc_given_decomp = 1.00` — once decomposition is right
   retrieval never misses, so the remaining ceiling is entirely here (67/161 correct
   at base). The one large unexplored axis.
2. Answer-accuracy leg for the resolver fix (`scripts/answer_accuracy_resolver.py`)
   — retrieval gains do not convert automatically: total-row injection converted
   on gpt-oss-120b and not at all on llama-3.1-8b.
3. Resolver fix on MultiHiertt / RealHiTBench (measured on HiTab only so far).
4. Row-axis reconstruction sits at .582 on real grids; 41% of tables encode the
   hierarchy in a single stub column, so the information is absent, not misread.

**Do not** add re-retrieval rungs, adjacency heuristics, or gate stages without
first designing the budget-matched control — all three have already been measured
as budget effects.
