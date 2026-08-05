# Operand-Set Completeness for Hierarchical-Table RAG

Masters-thesis research code. The working project is **[`rag-agent/`](rag-agent/)** —
see [`rag-agent/README.md`](rag-agent/README.md) for the code layout and how to run
the experiment legs, and [`rag-agent/RESEARCH_STRUCTURE.md`](rag-agent/RESEARCH_STRUCTURE.md)
for the claim-by-claim evidence audit.

**Rule this repo runs on:** *every number carries the results file it came from.*
A number without a source does not go in a document.

---

## The problem

An aggregation question over a hierarchical table — *"total revenue across all
divisions"* — needs **every** operand cell. Miss one and the sum is wrong. There
is no partial credit at the answer.

Retrieval is scored with per-fact recall, which cannot see this failure mode.
Recall@10 of 76% sounds fine and can still mean most aggregation queries are
missing at least one operand.

So the objective here is **Operand-Set Completeness (OSC)**:

> OSC(q) = 1 iff `gold_operands(q) ⊆ retrieved(q)`, else 0.

All-or-nothing, at the query level. Implementation: `rag_agent/eval/operand_set.py`.

---

## Positioning — read this before quoting anything as novel

Three things that look like contributions are not. They are stated here so nobody
in the lab builds on them by accident.

### ✗ "Verbalize each cell into a sentence, then embed and retrieve" — prior art

This is **MT2Net** (Zhao et al., *MultiHiertt*, ACL 2022, arXiv:2206.01347, §4),
verbatim:

> "we turn each cell into a sentence, along with its **hierarchical row and column
> headers**. For example, the first data cell … is translated as *'For Innovation
> Systems of Segment, sales of product in 2018, Year Ended December 31 is 2,894'*"

> "During the inference stage, the **top-n sentences are retrieved** as supporting facts."

That example sentence decomposes exactly as this repo's S3 template
(`For {row_path}, {col_path} is {value}`):

| fragment | role |
|---|---|
| `For Innovation Systems of Segment` | row path (Segment → Innovation Systems) |
| `sales of product in 2018, Year Ended December 31` | column path |
| `is 2,894` | value |

"Combining the row path and column path into a sentence" and "attaching hierarchical
row and column headers" are the same construction. The hierarchy source is not a
differentiator either — MT2Net states it uses "a **pre-processing script to extract
the hierarchical structure** of each HTML-format table", i.e. reconstructed, not
hand-annotated.

MT2Net also already published the flat-vs-hierarchical comparison:

> "Both TAGOP and FinQANet flatten each table by rows, losing the table's hierarchical
> structure information" … "they **ignore the table's hierarchical structure in the
> retrieving part**."

Same axis as flat vs S2/S3 here. Adopt and cite it; do not propose it.

Also prior art: **OHD** (arXiv:2602.01969) Table 2 ablation, markdown 53.33 → lineage
60.07 (+6.7pp) — the header-path serialization gain itself.

### ✗ "Fix MT2Net's 64% fact-integration error" — misreads the source

MT2Net's error analysis (Table 6, n=100 sampled errors) is a taxonomy of the
**generated program**, not an attribution of retrieval failure:

| category | share | paper's example |
|---|---|---|
| Wrong Operand or Span | 43% | `G: 327+415+1217` / `P: 426+517+1109` — *"Locate the wrong year"* |
| Missing Operand | 21% | `G: (1203+1437+1896+1774)/4` / `P: (1203+1774)/2` |
| Wrong Program | 19% | |
| Lack of Domain Knowledge | 4% | |

The paper never splits these into "retriever did not return the cell" vs "reasoner
did not use the returned cell". Quoting 43+21 = 64% as *retrieval* miss rate is a
misreading and does not survive one look at the table.

Our own measurements point the other way: whenever query decomposition is correct,
`osc_given_decomp = 1.00` (`results/e6_scope_treatments.json`, `results/e2_osc_enum*.json`).
The binding constraint is **which cells to ask for**, not whether retrieval can find them.

MT2Net is also a 2022 RoBERTa-scale model. Targeting its error rate in 2026 invites
"why not just use a current LLM?" as the first question.

### ✗ "Re-retrieve to recover missing operands" — measured, and it is a budget change

Two versions of this were built and both dissolve against a budget-matched control.

**Scope enumeration** — union the resolved header scope into top-k:
OSC .4252 → .5654 looks strong, but the evidence set grew from 10 to 19 cells.
Against a per-query budget-matched control (same cell count, filled by similarity
rank alone) the gain is **+.014, p=.55**. Dropped.

**Structural/subtree re-retrieval** — `results/e6_scope_treatments.json`:

| arm | OSC | mean cells | vs plain dense@10 |
|---|---|---|---|
| base | .4161 | 19.2 | −.373 |
| T_total_all | .5963 | **30.3** | −.193 |
| T_subtree | .4596 | — | — |

+.18 over base, with 1.6× the cells — and still far below plain dense@10.

**Completeness gate** — fires **0 times** at base k≥3 (`results/gate_k{1,2,3}.json`).
All 12 recoveries at k=1 came from the cheapest rung (a wider k); the structural and
enumeration rungs never fired at any k.

> **Standing rule from these failures:** any arm that grows the evidence set must
> report a per-query budget-matched control before its delta is quoted.

---

## What does hold up

### 1. OSC as the objective, and the failure modes only it can see

The literature scores per-fact recall and hands top-n to a reader. Set completeness
is not measured, so the conditions under which it breaks are not known. Measuring it
produces results that per-fact recall cannot express:

**Reranking raises per-item precision while breaking the set — under a narrow budget.**
`results/operand_collision_rerank_n300.json`, n=293 within-doc, exact McNemar:

| contrast | @10 | @20 | @50 |
|---|---|---|---|
| flat hybrid → rerank | 93→63, **p=5.2e-5** | 109→99, p=.184 | 141→142, **p=1.00** |
| S3 hybrid → rerank | 100→79, **p=7.5e-3** | 128→110, p=.015 | 163→155, p=.215 |

Independently confirmed by a pool-size sweep with gold always inside the pool
(`results/ea_pool_size_sweep_*.json`, `ea_crossover_stats.json`, Holm over family=18):
ΔOSC@10 is monotone in pool size and only the harmful end is significant —
S3 pool 500 p=.043, pool 1000 p=.0012; pool ≤200 indistinguishable from zero.

**The gain comes from ancestor header words, not from hierarchical order.**
Six shuffle draws of a length-matched control (`results/shuf_spread_s{1..5}.json`),
`set_em@10`:

| retriever | flat | S2_shuf mean (sd) | S2 | S2's rank |
|---|---|---|---|---|
| bm25 | .5529 | .5700 (.0000) | .5700 | 7/7 tie |
| dense | .5734 | .5819 (.0112) | .6041 | 1/7 |
| hybrid | .6075 | .6365 (.0064) | .6382 | 5/7 |
| cross | .5802 | .6348 (.0089) | .6553 | 1/7 |

Word effect (flat → shuf) is large and consistent. Order effect (shuf → S2) is
**not detectable** — hybrid's S2 sits inside the shuffle distribution. Best
permutation p ≈ 1/7 ≈ .14, and the sign reverses at @50. No order claim is made.
bm25's exact 0 is bag-of-words by construction, not a measurement.

### 2. Header-path serialization beats flat, against a real baseline

RealHiTBench, raw hierarchical tables, strict `em_norm`, same retriever and k —
only the serialization differs (`results/realhitbench_s1_vs_s2*`):

| solver | n | S1 flat | S2 header-path | Δ | p |
|---|---|---|---|---|---|
| gpt-5.1 | 94 | .160 | .309 | +.149 | .0043 |
| gpt-4.1-mini | 94 | .170 | .277 | +.106 | .0213 |
| llama-3.3-70b | 44 | .205 | .341 | +.136 | .0703 |

Same direction across three solver families. (Not comparable to the RealHiTBench
leaderboard — that uses an LLM judge, this a local scorer.)

### 3. Semantic header-path resolution — the one intervention that passes budget matching

The lexical scorer in `resolve_against_table` keeps only header paths with a nonzero
token overlap, discarding **28.0% of gold rows and 31.3% of gold columns** before
ranking. Nothing is structurally unreachable; every gold row/column has a real
non-empty header path. Wiring `EmbedResolver` in lifts both-axes resolution
.285 → .537 (p=1.8e-11) and OSC — **at a matched cell count**
(`results/resolver_osc_matched_train.json`, HiTab train, n=1,006):

| k | lexical (budget-matched) | semantic | Δ | p |
|---|---|---|---|---|
| 5 | .4513 | .5000 | +.049 | 8.1e-4 |
| 10 | .5656 | .6034 | +.038 | 6.2e-3 |
| 20 | .6948 | .7346 | +.040 | 1.6e-3 |

This is not "retrieve more" — it is choosing which header path to target, at the same
budget. Dev split: `results/resolver_osc_matched_dev.json`.

### 4. Structural total-row injection — a HiTab-scoped case study, not the headline

Index-time, language-independent detection of aggregate rows; zero per-query cost.

⚠️ **The population-lift claim does not hold.** Injection changes retrieval (OSC 0→1)
on only **10 of 161** queries (6.2%); the rest are structurally destined to tie.
`--flips-first` put those 10 at the front, so a run cut short by a token quota was
scored on a flip-enriched sample. Scoring the completed records at successive cutoffs
(official `hitab_exact_match`, McNemar):

| N scored | flip share | gpt-oss-120b Δ (p) | gpt-4o Δ (p) |
|---|---|---|---|
| 86 | 11.6% | +.093 (.039) | +.047 (.219) |
| 104 | 9.6% | +.077 (.039) | +.019 (.727) |
| 140 | 7.1% | +.036 (.332) | +.014 (.754) |
| 161 | 6.2% | 21 pending | +.019 (.581) |

Significance exists only at N≤104. **Do not cite any truncated-sample p-value.**

What survives is conditional transfer — on the 10 flipped queries, accuracy goes
.00 → **.90** (gpt-oss-120b, 9:0) and .00 → **.50** (gpt-4o, 5:0), with zero losses.
Retrieval fixes do convert; the arithmetic ceiling on population lift is
6.2% × .90 ≈ **5.6pp**. Also HiTab-only: WikiSQL has no aggregate cells, FinQA tables
are too small (median 5 rows).

---

## How to state the contribution

Not "we verbalize cells and retrieve" (2022), not "we fix a 64% error rate"
(misread), not "we re-retrieve to recover misses" (a budget change):

> Aggregation queries over hierarchical tables fail all-or-nothing — one missing
> operand cell makes the answer definitively wrong. Existing table QA hands top-n
> facts to a reader and scores the answer, so this set property is invisible to its
> metrics. Scoring **Operand-Set Completeness** instead reveals (i) the conditions
> under which completeness breaks — reranking trades set completeness for per-item
> precision, and only under a narrow budget; the gain from header paths is lexical,
> not ordinal; (ii) that the ceiling is **query decomposition**, not retrieval
> (`osc_given_decomp = 1.00`); and (iii) that the intervention which survives a
> per-query budget-matched control is semantic header-path targeting, not re-retrieval.

The novelty is the metric plus the diagnoses it makes visible. The negative results
are part of the contribution: each one is a plausible mechanism that a properly
matched control kills.

---

## Status

Current handoff note: [`rag-agent/STATUS-2026-08-06.md`](rag-agent/STATUS-2026-08-06.md).

Other directories (`T2/`, `hart-table-retrieval/`) are earlier / adjacent work and are
not part of the thesis line above.
