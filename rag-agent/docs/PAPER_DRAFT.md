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
2. **Header-tree scope enumeration** — a header node *is* an aggregation scope, so its
   operand set is **complete-by-construction**; this is **scope-size robust** where
   similarity retrieval collapses, and re-localizes the problem to header-path
   decomposition.
3. **Both axes are node resolution, and a cross-encoder wins both** — picking the
   right scope node (query × header joint attention) beats lexical and bi-encoder
   matchers on **col-recall@2 0.40→0.70** and **row-recall@2 0.44→0.52 (p<0.01)**.
   Plus axis-specific diagnosis: unnamed *total* rows (68% of row failures) fixed by
   total-row augmentation (row-cov 0.62→0.89).
4. **Honest frontier + negative results** — completeness vs precision is a frontier
   ("100% in a small set" open); several intuitive heuristics are shown *not* to help.

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
- **Metrics.** OSC (primary); **col-recall@k** / **row-recall@k** (gold columns/rows
  within the top-k scope-nodes — the node-resolution metric per axis); row-/col-axis
  coverage; mean cells
  (precision); answer accuracy (numeric-match + exact-match) for the generation stage.
- **Models.** Embedder `BAAI/bge-small-en-v1.5`; cross-encoders
  `ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`; solver Groq
  `llama-3.1-8b-instant` (codegen) — all retrieval is LLM-free; only the answer stage
  uses an LLM.
- **Protocol.** Paired comparisons; bootstrap 95% CI + McNemar; "accuracy over
  answered" excludes failed/oversize LLM calls. Reproduce: `scripts/e1..e7`,
  `diag_row_failures.py`, `col_select_bench.py`, `row_select_bench.py`,
  `row_select_stats.py`, `row_osc_endtoend.py`, `dense_ceiling_diag.py`,
  `retrieval_stage_eval.py`.

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
| share of gold operands | **28.5%** (140/491) | 71.5% |
| median similarity rank | **39.5** | 8 |
| reached within top-50 | **0.593** | 0.906 |

44.7% of queries need ≥1 total-row operand; similarity ranks those cells ~5× worse, so
**40% are still unreached at k=50**. Dense full-set completeness plateaus accordingly
(@10 0.366 → @50 0.714), and **76% (35/46) of the @50 incompletes are explained by an
unreached total-row operand**. The miss is *structural, not a budget problem*: these
cells resemble the query neither semantically (dense) nor lexically (BM25), so **no
similarity/hybrid retriever reaches them by construction** — header-tree enumeration
does, because a total row falls under the scope node regardless of resemblance. (This is
the mechanism behind the completeness guarantee; `dense_ceiling_diag`.) *Caveat:
ordinary operands also plateau below 1.0 (0.906), so total rows are the largest but not
the only cause; `is_total_row` is a heuristic (empty/"total"/"overall" paths).*

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

**Honest position.** We do **not** claim to retrieve operand cells *more often* than
similarity retrieval — on raw average OSC, dense top-k beats enumeration (0.79 > 0.65).
That is not the contribution and we do not hide it. The contribution is about a
*different objective*: aggregation needs the **complete** operand set, and we show
(§5.1b) that **similarity retrieval cannot satisfy that objective by construction** —
28.5% of operands are structurally-required total rows it ranks ~5× worse and misses
even at k=50, explaining 76% of its completeness ceiling. Against that, our contribution
is: (i) **OSC** as the all-or-nothing completeness objective existing relevance/ranking
retrievers (incl. 2026 cell-level table RAG: FT-RAG, Topo-RAG — partial recall / nDCG)
do not target; (ii) **header-tree enumeration**, complete-by-construction and
scope-robust, that reaches the cells similarity cannot; (iii) a **cross-encoder
node-resolution** result on *both* axes (significant on row-recall p=0.007 and
col-recall; column generalizing to AITQA) — a real retriever improvement, though on the
row axis its end-to-end OSC lift is only directional (+0.05, p=0.08), bounded by the
column-axis ceiling; (iv) **scalability** — 9× fewer tokens, feasible where whole-table
serialization is not (§5.9). We are explicit that the proposed method trades
average-recall for a completeness *guarantee* + efficiency, and that closing the raw-OSC
gap (the query→node decomposition bottleneck) remains open.

### Open / limitations
- Column completeness on two-entity comparisons (~25% residual) — needs heavier
  methods (LLM/fine-tuned schema linker); future work.
- End-to-end answer accuracy is solver-limited (8b); a stronger solver is needed.
- Single dataset (HiTab); selection/comparison aggregations excluded.
