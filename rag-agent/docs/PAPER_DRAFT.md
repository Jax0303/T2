# Operand-Set Completeness: Header-Tree Retrieval for Aggregation over Hierarchical Tables — draft

All numbers: HiTab `dev`, arithmetic aggregations, distinct-cell scope **m≥2 (n=161)**,
current gold, seed 42, paired where applicable. LLM-free unless noted.

---

## 1. 개요 (Introduction)

Hierarchical tables (statistical reports, finance) ask aggregation questions whose
answer is computed from **multiple operand cells**. Getting the answer right requires
retrieving **every** operand — a strict *completeness* requirement that ordinary RAG,
which ranks passages by *relevance*, does not target: as the aggregation scope grows,
similarity retrieval drops operands.

We reframe the task at the **retrieval stage**: retrieve the complete operand set with
as few cells as possible. Contributions:
1. **Operand-Set Completeness (OSC)** — an all-or-nothing retrieval objective/metric
   for table aggregation (a 2025 TQA survey confirms this is unmeasured).
2. **A structural ceiling diagnosis of similarity retrieval, not just a benchmark
   number.** We show *why* BM25/dense/hybrid plateau below full completeness: unnamed
   total/aggregate rows share neither vocabulary nor semantics with the query, so they
   rank ~4× worse and are missed even at k=50 — this single cause explains **62%** of
   similarity retrieval's completeness ceiling (§5.1b). This is a general, dataset-level
   claim about *why* relevance-ranked retrieval cannot satisfy an all-or-nothing
   objective by construction, independent of any fix we propose for it.
3. **Header-tree scope enumeration** — a header node *is* an aggregation scope, so its
   operand set is **complete-by-construction**; this is **scope-size robust** where
   similarity retrieval collapses, and re-localizes the problem to header-path
   decomposition.
4. **Both axes are node resolution, and a cross-encoder wins both** — picking the
   right scope node (query × header joint attention) beats lexical and bi-encoder
   matchers on **col-recall@2 0.40→0.70** and **row-recall@2 0.44→0.52 (p<0.01)**.
   Plus axis-specific diagnosis: unnamed *total* rows (68% of row failures) fixed by
   total-row augmentation (row-cov 0.62→0.89).
5. **Case study: the ceiling diagnosis (2) is actionable, not just descriptive.** As a
   worked proof-of-concept, not a proposed general method, we hand-inject total rows for
   the subset of queries whose gold operands actually need one (35% of the population)
   and show BM25 and RRF-hybrid rise significantly there (p≤0.001, closing 58–65% of
   their completeness gap; dense directionally, p=.06) with zero regression elsewhere. We are explicit that this patch is narrow on three axes — query
   type (only total/ratio queries), detection mechanism (an English keyword heuristic,
   not a learned one), and dataset (HiTab-specific; §5.11) — and report it as evidence
   the diagnosis has teeth, not as a general retrieval improvement.
6. **Honest frontier + negative results** — enumeration *alone* does not beat dense on
   raw OSC (the win is by augmentation, not replacement); completeness vs precision is a
   frontier ("100% in a small set" open); several intuitive heuristics *do not* help.

## 2. 관련 연구 (Related Work)

- **Table QA / parsers (HiTab, MAPO+TaBERT, 45.5% dev).** End-to-end denotation
  accuracy with a 2022 BERT-era parser; `linked_cells` are *supervision*, not a
  minimal-complete retrieval target.
- **Table-RAG / cell selection (TableRAG, H-STAR, Chain-of-Table).** Optimize answer
  accuracy / token-efficiency on **flat** tables; report graceful recall, **no
  completeness guarantee**; aggregation often deferred to SQL.
- **Hierarchical representation (OHD 2602.01969, HD-RAG).** Build the *same* orthogonal
  row/column header trees on HiTab, but to **serialize the whole table** for the LLM
  (no retrieval/selection). Complementary to us (representation vs retrieval); we do
  **not** claim the tree representation or the (header-path=value) format as novel.
- **Schema linking (CE-SL, RESDSQL).** Map a question to relevant **columns** with a
  **cross-encoder** — we adopt this for our column axis.
- **Gap.** No prior work makes header-tree scope enumeration / operand-set
  completeness a retrieval-time objective for hierarchical-table aggregation; evidence-
  cell completeness for aggregation is an unmeasured area (survey).
- Detail: `RELATED_DELTA.md`, `related_methods_study.md`, `retrieval_algorithms_study.md`.

## 3. 방법론 (Methodology)

- **3.1 OSC.** Given query *q*, hierarchical table *T* (top/left header trees), gold
  operand set *G*: **OSC(q)=1 iff G ⊆ retrieved** (all-or-nothing subset containment) —
  the necessary condition for a correct aggregation, strictly harder than mean cell
  recall. (`eval/operand_set.py`)
- **3.2 Header-tree scope enumeration.** Resolve *q* to header-path predicates, then
  enumerate every numeric leaf under the matched row × column scope nodes
  (`retrieve/header_enum.py`). Complete-by-construction: if the scope node is correct,
  OSC=1 regardless of scope size.
- **3.3 Query→node decomposition.**
  - *Row axis* — **cross-encoder** reranking of (query, row-header node)
    (`query/header_embed_resolver.py`, `row_mode="cross"`): beats the prior embedding
    tree-node default on row-recall at every budget (§5.3), symmetric with the column
    axis. The embedding matcher (semantic node match — closes vocabulary/depth gaps,
    matches a 70b LLM, LLM-free) remains the fallback.
  - *Column axis* — **cross-encoder** reranking of (query, column-header) — schema-
    linking SOTA; cascade = lexical first, cross-encoder when lexical finds nothing.
- **3.4 Diagnosis-driven augmentation.** Ratio/share queries need an *unnamed* total
  row (the denominator): `total_like_rows` detects table/section totals (empty or
  "total"/"overall" header) and unions them in; `expand_sibling_groups` completes a
  partially matched sibling set. (`header_enum.py`)
- **3.5 Recall-first guarantee.** When 100% completeness is mandatory, union/fallback
  to a provably-complete set (axis-complete ∪ dense, or whole table) and minimize under
  the constraint — completeness is guaranteed structurally, precision is the objective.

## 4. 실험 세팅 (Experimental Setup)

- **Data.** HiTab dev; gold operands resolved from `linked_cells.quantity_link` by
  value-matching (`bench/hitab.py`). Population: arithmetic aggregations with scope
  **m≥2, n=161**; selection/comparison excluded (value-matching cannot build gold).
- **Metrics.** OSC (primary); **mean per-cell recall** reported alongside OSC in the
  main comparisons (the graceful metric reference table-RAG systems report — the
  OSC↔recall gap is itself an exhibit: partial recall is not completeness);
  **col-recall@k** / **row-recall@k** (gold columns/rows
  within the top-k scope-nodes — the node-resolution metric per axis); row-/col-axis
  coverage; mean cells
  (precision); answer accuracy for the generation stage — cross-paper numbers use
  HiTab's own scorer (`hitab_exact_match`); our lenient `numeric_match` is
  diagnostic-only. Table-level retrieval is additionally benchmarked on standard IR
  metrics (R@1/5/10, MRR, nDCG@10; `multidataset_retrieval`).
- **Models.** Embedder `BAAI/bge-small-en-v1.5`; cross-encoders
  `ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`; solver Groq
  `llama-3.1-8b-instant` (codegen) — all retrieval is LLM-free; only the answer stage
  uses an LLM.
- **Protocol.** Paired comparisons; bootstrap 95% CI + McNemar; "accuracy over
  answered" excludes failed/oversize LLM calls. Reproduce: `scripts/e1..e7`,
  `diag_row_failures.py`, `col_select_bench.py`, `row_select_bench.py`,
  `row_select_stats.py`, `row_osc_endtoend.py`, `dense_ceiling_diag.py`,
  `osc_total_augment.py`, `retrieval_stage_eval.py`.

## 5. 결과 (Results)

**5.1 Similarity retrieval's OSC collapses with scope (H1).** Dense single-vector OSC
falls as the aggregation scope m grows (m=2→0.60 … m=9+→0.29); completeness is bought
with budget, not targeting. (E1)

**5.1b Why similarity retrieval has a completeness ceiling it cannot pass (the
mechanism).** Aggregation operands include cells the query does *not* resemble — chiefly
the **unnamed total row** (the share/ratio denominator), whose header is empty or just
"total". Ranking every numeric cell by query↔header-lineage cosine (HiTab dev arith
m≥2, n=161, LLM-free):

| | total-row operand | ordinary operand |
|---|---|---|
| share of gold operands | **13.4%** (66/491) | 86.6% |
| median similarity rank | **32.0** | 8 |
| reached within top-50 | **0.652** | 0.897 |

29.8% of queries need ≥1 total-row operand; similarity ranks those cells ~4× worse, so
**35% are still unreached at k=50**. Dense full-set completeness plateaus accordingly
(@10 0.429 → @50 0.770), and **62% (23/37) of the @50 incompletes are explained by an
unreached total-row operand**. The miss is *structural, not a budget problem*: these
cells resemble the query neither semantically (dense) nor lexically (BM25), so **no
similarity/hybrid retriever reaches them by construction** — header-tree enumeration
does, because a total row falls under the scope node regardless of resemblance. (This is
the mechanism behind the completeness guarantee; `dense_ceiling_diag`.) *Caveat:
ordinary operands also plateau below 1.0 (0.897), so total rows are the largest but not
the only cause; `is_total_row` is a heuristic (empty/"total"/"overall" paths). Prepending
the table title to each cell's embedded text does **not** rescue total-row operands —
median rank *worsens* (32.0→36.5) and reachable@50 drops (0.652→0.576), because
statistical-report titles reuse similar phrasing across tables and dilute the
header-path signal rather than adding table-specific discriminating signal
(`dense_ceiling_diag --with-title`, `results/dense_ceiling_diag_with_title.json`). This
closes the "just add the caption" objection.*

**5.2 Enumeration is scope-robust and re-localizes the bottleneck (H2).** OSC |
decomposition-correct = **1.000, flat across m**; the H1 collapse is eliminated. Raw
OSC equals the decomposition success rate, so the bottleneck is **header-path
decomposition**, localized to the row axis. (E2)

**5.3 Row axis — node resolution + total-row augmentation.** Like the column axis,
picking the right row scope-node is a node-resolution problem; **row-recall@k** (gold
rows within the rows covered by the top-k row scope-nodes) compares matchers fairly:

| row selector | @1 | @2 | @3 | @4 |
|---|---|---|---|---|
| lexical | 0.193 | 0.335 | 0.398 | 0.441 |
| bi-encoder (prior default) | 0.267 | 0.435 | 0.578 | 0.615 |
| cross-encoder (MiniLM) | **0.311** | **0.522** | 0.602 | 0.665 |
| cross-encoder (bge-reranker) | 0.298 | 0.503 | **0.609** | **0.671** |

A cross-encoder beats the embedding default at **every** k (paired McNemar @2: 19
cross-only vs 5 embed-only, **p=0.007**) — the same ordering the column axis gives.
End-to-end (`row_mode` embed→cross, column axis held fixed at lexical) this converts
to row-cov 0.615→0.665 and **ΔOSC +0.050** (paired CI [0.00, 0.099]; full-OSC McNemar
12 cross-only vs 4 embed-only, p=0.08) — a real-direction but **borderline** lift,
capped by the column-axis ceiling (OSC requires *both* axes correct). The row-recall
gain transfers 1:1 to row coverage; its OSC payoff is bounded, not free.
(row_select_bench, row_select_stats, row_osc_endtoend)

Orthogonally, **68%** of the residual row-axis misses are share/ratio queries needing
an *unnamed* total row; total-row augmentation lifts **row-cov 0.615 → 0.888** and OSC
0.416 → 0.652 (paired ΔOSC **+0.236**, CI [0.174, 0.304]) — the larger row lever, and
complementary (it fixes missing *totals*, the cross-encoder fixes *named-row* matching).
(diag, E6)

**5.4 Column axis as schema linking.** OSC rewards the trivial whole-axis dump, so we
measure **col-recall@k**. A cross-encoder is the best selector at every budget:

| column selector | @1 | @2 | @3 |
|---|---|---|---|
| lexical (prior default) | 0.267 | 0.398 | 0.472 |
| bi-encoder | 0.267 | 0.565 | 0.677 |
| cross-encoder (MiniLM) | 0.398 | 0.609 | 0.727 |
| **cross-encoder (bge-reranker)** | 0.373 | **0.702** | **0.795** |

+0.30 @2 over lexical. Residual (cross top-3 fails, n=44): 77% two-column comparisons,
18% multi-column, ~2 year-ambiguous. (col_select_bench)

**5.5 Completeness↔precision frontier.** precise enum 0.42@19c · treated 0.65@40c ·
dense top-10 0.79@57c · whole table 1.00@160c; recall-first union = **1.00@~123c**.
100% is reachable but costs ~76% of the table; "100% in a small set" is open. (E5)

**5.6 Generation stage.** With retrieval fixed at oracle, **structured (header-path =
value) context** raises numeric-match 0.34→0.58 and cuts silent errors 0.66→0.42 (E4).
At a fixed solver, an 8b model **floors every retrieval arm at ~0.13** while oracle
(3 cells) reaches 0.61 — **answer accuracy is precision-dominated and solver-limited**;
retrieval gains need a stronger solver to surface end-to-end. (E7)

**5.7 Negative results (ablations).** Last-column default, named-pair query
decomposition, and total-**column** augmentation each **fail to beat** the simple
strong-cross-encoder top-2 (e.g. col-recall on comparison queries: blind top-2 0.75 vs
named-pair 0.53 vs total-col 0.48). The strong cross-encoder at a small budget is the
right column method; clever decomposition does not help.

**5.8 External validity — column selection on AITQA (2nd dataset).** AITQA (airline
SEC filings, hierarchical headers) gives no operand labels, so OSC is not computable;
but answer-value matching recovers a unique gold **column** for 439 questions, letting
us re-run the column-selection benchmark on a new domain:

| selector | col-recall@1 | @2 | @3 |
|---|---|---|---|
| lexical | 0.795 | 0.913 | 0.973 |
| bi-encoder | 0.754 | 0.886 | 0.968 |
| **cross-encoder** | **0.820** | **0.923** | **0.979** |

The ordering **generalizes** — cross-encoder best, bi-encoder worst, at every k — so
"use a cross-encoder for the column axis" holds across domains. The *margin* is small
here because AITQA columns are clean metric names (small vocabulary gap); the large
cross-encoder gain on HiTab is specific to harder gaps ("%"↔"percentage"). OSC
external validity remains a limitation (no operand labels outside HiTab).

**5.9 vs OHD-style whole-table serialization — retrieval is feasible at a fraction of
the context.** OHD (2602.01969) builds the *same* orthogonal header trees but
serializes the **whole** table for the LLM (no selection). `ohd_lite` reproduces its
dual (row- + column-major) `Context→Key→Value` rendering inside our harness
(`e7_retrieval_ablation.py::ohd_serialize`; omits OHD's learned tree induction +
semantic arbitrator). LLM-free context cost (HiTab dev arith m≥2, n=161):

| arm | mean tokens | oversize @8k ctx | tokens vs ohd_lite |
|---|---|---|---|
| ohd_lite (whole table) | 8,518 | 56/161 (**35%**) | 1× |
| dense top-10 | 1,502 | 0 | 5.7× fewer |
| **enum_treated (ours)** | **953** | **0** | **8.9× fewer** |
| **enum_cross (ours)** | **753** | **0** | **11.3× fewer** |

Whole-table serialization **exceeds an 8k context on 35%** of tables (and uses ~9× the
tokens where it fits); every retrieval arm runs on **100%** — feasibility, not just
thrift. *(H6 accuracy-parity at a fixed 70b solver — that the token saving is free, not
paid in accuracy — pending; reads on the small-table subset where ohd_lite fits.)*
(E8, `e8_scalability_dryrun` / `e8_ohd_baseline`)

**5.9b Same-metric head-to-head: OSC at a fixed token budget.** E8 shows whole-table
serialization is *expensive*; E9 shows it is also **less complete at every budget that
matters**. LLM-free, identical per-cell rendering and population for all arms;
retrieval arms take the largest k that fits B (no gold peeking); OHD arms get both a
*strict* variant (whole table or nothing — OHD has no selection mechanism) and a
*generous* row-major-prefix truncation it does not actually have. Whole-table cost:
mean 4,382 tokens (p90 12,332; OHD's faithful dual serialization doubles it).
OSC@B (re-measured post-audit 2026-07-08):

| B (tokens) | 1000 | 2000 | 4000 | 8000 | 16000 |
|---|---|---|---|---|---|
| **hybrid+inject (ours)** | **0.764** | **0.876** | **0.963** | **1.000** | 1.000 |
| dense plain | 0.721 | 0.851 | 0.944 | 0.981 | 1.000 |
| ohd_trunc (generous) | 0.553 | 0.671 | 0.807 | 0.870 | 1.000 |
| ohd_dual_strict (faithful) | 0.050 | 0.273 | 0.416 | 0.652 | 0.795 |

Paired vs the *generous* OHD arm, injection wins at **every** B in 250–8k (Δ +0.12 to
+0.21, McNemar p≤0.004, at B≥2k flips ≥21 vs ≤4); vs faithful OHD the gap is ~0.5 OSC.
Whole-table only reaches parity at **B≥16k** — precisely the regime where selection is
unnecessary, matching the generalization study's scope claim (§5.11 / FinQA). At 8k
tokens ours delivers **OSC=1.000** while whole-table still fails 13% of queries.
**Mean per-cell recall — the metric reference systems report — hides this ordering:**
at B=2k hybrid-plain scores recall 0.92 while completing only 0.84 of queries, and
ohd_trunc@8k reports recall 0.92 with 13% of queries uncomputable; at 4k every strong
arm sits in a 0.96–0.98 recall band while OSC spans 0.81–0.98. Partial recall's
per-cell credit is exactly what an all-or-nothing aggregation cannot spend.
*Honest caveat:* at starvation budgets (≤500 tokens) injected total cells crowd out
ranked chunks and plain dense is best (@500: 0.497 plain vs 0.441 inject); the patch
needs ~1k tokens of headroom (crossover ≈1k). (E9, `e9_osc_token_budget`)

**5.10 Case study: injecting total rows converts the diagnosis (2) into a measurable,
significant fix — for the queries it applies to.** This section is **not** proposed as a
general retriever improvement (see the scoping in Contribution 5): it is a worked
demonstration that the ceiling diagnosis is actionable. Using the cross-encoder column
resolver (§5.4's winner, bge-reranker) to pick 1–2 columns, we union *only those
columns'* total-like rows (mean 4.4 cells/query) into any retriever's top-k.
**Total-row detection is now a hybrid
of the keyword heuristic and a language-independent structural check** — a row is also
flagged if it arithmetically sums its tree children or row-level siblings, regardless of
its text label (`is_total_row_structural`; catches e.g. an unlabeled "federal government"
subtotal the keyword regex misses). Structural detection *alone* under-performs keyword
alone (it tends to (re)find totals similarity retrieval could already reach); the union
is never worse than keyword alone and widens applicability. At the **same retrieval
depth** (k=10; HiTab dev arith m≥2, n=161; re-measured post-audit 2026-07-08), with
mean per-cell recall — the reference metric — alongside:

| baseline | plain OSC | +injection | Δ | flipped | hurt | McNemar p | recall plain→aug |
|---|---|---|---|---|---|---|---|
| BM25 | 0.770 | **0.876** | +0.106 | 17 | **0** | **1.5e-5** | 0.887→0.945 |
| dense | 0.832 | 0.863 | +0.031 | 5 | **0** | 0.0625 (n.s.) | 0.920→0.946 |
| hybrid | 0.839 | **0.907** | +0.068 | 11 | **0** | **0.001** | 0.927→0.964 |

Injection only *adds* cells, so it is a strict superset — **zero queries are hurt** — but
dense's same-depth win is not clean (p=0.0625) at this population size; BM25 and
RRF-hybrid are significant. Note the recall column: plain retrievers already score
0.89–0.93 on averaged per-cell recall while completing only 0.77–0.84 of queries —
the graceful metric conceals exactly the failures injection targets. As before, the
population-average Δ is diluted by
construction: only **35% of queries (56/161) have a gold operand that is a total-like
row** — for the other 65%, injection is a structural no-op (Δ=0.000, p=1.0 for all
three). Splitting by whether the query actually needs a total row:

| baseline | subset | plain OSC | +injection | Δ | gap closed | McNemar p |
|---|---|---|---|---|---|---|
| BM25 | needs total (n=56) | 0.536 | **0.839** | +0.304 | **65%** | **2e-5** |
| dense | needs total (n=56) | 0.714 | **0.804** | +0.089 | 31% | 0.0625 (n.s.) |
| hybrid | needs total (n=56) | 0.661 | **0.857** | +0.196 | **58%** | **0.001** |
| any | doesn't need total (n=105) | — | — | 0.000 | — | 1.0 |

Read correctly, the result is: *for the diagnosed failure mode, the patch closes 31–65%
of the remaining completeness gap with zero collateral damage elsewhere, for BM25 and
RRF-hybrid at conventional significance* — not "a general +3-10pp retrieval improvement,"
and dense specifically is a directional-but-not-significant win at this sample size.
*Honest budget caveats:* (i) under a deeper-budget comparison (aug@10, ~62 cells, vs
plain@20, ~98 cells) plain@20 **significantly beats** aug@10 for dense (0.863 vs 0.950,
p=0.001) and hybrid (0.907 vs 0.975, p=0.013), with BM25 at parity (0.876 vs 0.882,
p=1.0) — simply fetching more rows is a real competitor once the baseline itself is
strong, and this comparison does not use equal cell counts either way (aug@10 is
cheaper). (ii) Under a strict per-query *cell-matched* budget (plain deepened until it
holds ≥ aug's cell count) only BM25's win survives (+0.106, p<1e-4; dense and hybrid
+0.019, n.s.) — the post-audit baselines are strong enough that matched-budget claims
must be scoped to BM25. The clean three-retriever-direction win is at *matched depth*,
not matched budget. (`osc_total_augment`)

**Answer-stage payoff (H6, preliminary — quota-limited sample).** Does the completeness
gain convert to final answer accuracy? At a fixed capable solver (gpt-oss-120b, codegen,
official HiTab EM), injecting total rows into dense@10 lifts EM **0.395→0.500** on the
86/161 queries evaluated so far (paired; 11 wrong→right vs 2 right→wrong, McNemar
**p=0.022**); on the 11 queries whose OSC flips 0→1, accuracy goes **0.00→0.73**. A
70B solver is directional but underpowered (0.346→0.404, 3:0, p=0.25, n=52) and an 8B
solver does not move at all (p=1.0, n=134) — completeness gains surface only above a
solver-capability floor, consistent with §5.6's precision-dominated reading. *Caveats:*
the sample is quota-cut and flips-first-ordered (queries injection can affect are
evaluated first), so the evaluated-subset Δ overstates the population Δ — significance
rests on the paired flip counts, not the means; 4/86 treatment contexts hit the context
budget (truncation detected and counted). (`answer_accuracy_injection`,
`results/h6_rerun_20260707/`)

**5.11 Generalization scope: where the injection win does — and cannot — transfer.**
Applying the same pipeline to FinQA and WikiSQL (gold-operand m≥2 populations) shows
the win is **regime-specific for two distinct structural reasons**, which sharpens
rather than weakens the claim. *WikiSQL (flat relational): mechanism inapplicable* —
**0%** of gold tables contain a named total row (aggregates are computed on the fly,
never stored), so there is nothing to inject. *FinQA (small financial tables, median
5 rows): mechanism unnecessary* — total rows exist (39%) but any budget k≥10 already
retrieves the whole table (mean 0.1 cells injected, ΔOSC=0); its ceiling (OSC 0.315
*with the full table in context*) is operand decomposition, not reachability. HiTab is
the regime with *both* preconditions: tables too large for a realistic budget **and**
stored, structurally-dissimilar total rows. The contribution is therefore precisely
scoped: *retriever-agnostic completeness patching for large hierarchical tables with
stored aggregates* — consistent with §5.9b, where whole-table serialization reaches
parity exactly when tables fit the budget. (`finqa_total_inject`,
`total_inject_generalization_summary`)

**Honest position.** *Enumeration alone* does **not** retrieve operand cells more often
than similarity retrieval — on raw average OSC, dense beats our header-tree enumeration
(dense@10 now 0.83 post-bugfix, §5.1b; the enumeration-side number predates this
session's gold/row-path fixes and needs re-verification before being re-quoted — see
`codebase-audit-bugfixes-2026-07-07` — omitted here rather than re-stating a stale
figure). The contribution is about a *different objective*:
aggregation needs the **complete** operand set, and we show (§5.1b) that **similarity
retrieval cannot satisfy that objective by construction** — 13.4% of operands are
structurally-required total rows it ranks ~4× worse and misses even at k=50, explaining
62% of its completeness ceiling. This diagnosis — general, and independent of any fix —
is the primary claim; ranked below it, in order of how general each result actually is:
(i) **OSC** as the all-or-nothing completeness objective existing relevance/ranking
retrievers (incl. 2026 cell-level table RAG: FT-RAG, Topo-RAG — partial recall / nDCG)
do not target; (ii) the **ceiling diagnosis** itself (above); (iii) **header-tree
enumeration**, complete-by-construction and scope-robust, that reaches the cells
similarity cannot — though it does not beat baseline OSC on its own; (iv) a
**cross-encoder node-resolution** result on *both* axes (significant on row-recall
p=0.007 and col-recall; column generalizing to AITQA) — a real retriever improvement,
though on the row axis its end-to-end OSC lift is only directional (+0.05, p=0.08),
bounded by the column-axis ceiling; (v) **scalability** — 9× fewer tokens, feasible
where whole-table serialization is not (§5.9). The **total-row injection case study**
(§5.10) is deliberately *not* listed here as a general contribution — it demonstrates
(ii) is actionable within a narrow, explicitly-scoped slice (35% of queries, HiTab-only;
see limitations), and should be read as a worked example, not as evidence of a general
retrieval improvement.

### Open / limitations
- **The total-row injection case study (§5.10) is narrow on two remaining axes** (a
  third — detection mechanism — is now partially addressed, see below): query type
  (only the 35% of queries whose gold needs a total/ratio row — no effect, positive or
  negative, on the rest) and dataset (HiTab-specific — §5.11 shows it is inapplicable to
  WikiSQL and unnecessary for FinQA).
- **Detection mechanism, partially fixed.** Total-row detection was a pure English
  keyword regex ("total"/"overall"/"all"); it is now a hybrid of that regex and a
  language-independent structural check (row value arithmetically sums its tree children
  or row-level siblings — `is_total_row_structural`). This is a genuine generality gain
  (works without any English cue) but structural detection *alone* under-performs the
  keyword heuristic on this dataset (§5.10) — it tends to rediscover totals similarity
  retrieval could already reach rather than the vocabulary-dissimilar ones that are
  actually hard — so the current hybrid still leans on the keyword signal for its
  effect size. Testing the structural-only detector on a non-English hierarchical-table
  benchmark (none identified yet) would be the real test of the generality claim.
- Column completeness on two-entity comparisons (~25% residual) — needs heavier
  methods (LLM/fine-tuned schema linker); future work.
- End-to-end answer accuracy is solver-limited (8b); a stronger solver is needed.
- Single dataset (HiTab); selection/comparison aggregations excluded.
