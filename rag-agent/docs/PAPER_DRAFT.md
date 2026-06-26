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
3. **Axis-wise diagnosis + treatments** — row axis: unnamed *total* rows (68% of
   failures) fixed by total-row augmentation (row-cov 0.62→0.89); column axis cast as
   **schema linking**, solved best by a **cross-encoder** (col-recall@2 0.40→0.70).
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
  - *Row axis* — **embedding tree-node resolver** (`query/header_embed_resolver.py`):
    match the query to header nodes by semantic embedding (closes vocabulary/depth
    gaps; matches a 70b LLM, LLM-free).
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
- **Metrics.** OSC (primary); **col-recall@k** (gold columns within top-k — the
  schema-linking metric for the column axis); row-/col-axis coverage; mean cells
  (precision); answer accuracy (numeric-match + exact-match) for the generation stage.
- **Models.** Embedder `BAAI/bge-small-en-v1.5`; cross-encoders
  `ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`; solver Groq
  `llama-3.1-8b-instant` (codegen) — all retrieval is LLM-free; only the answer stage
  uses an LLM.
- **Protocol.** Paired comparisons; bootstrap 95% CI + McNemar; "accuracy over
  answered" excludes failed/oversize LLM calls. Reproduce: `scripts/e1..e7`,
  `diag_row_failures.py`, `col_select_bench.py`, `retrieval_stage_eval.py`.

## 5. 결과 (Results)

**5.1 Similarity retrieval's OSC collapses with scope (H1).** Dense single-vector OSC
falls as the aggregation scope m grows (m=2→0.60 … m=9+→0.29); completeness is bought
with budget, not targeting. (E1)

**5.2 Enumeration is scope-robust and re-localizes the bottleneck (H2).** OSC |
decomposition-correct = **1.000, flat across m**; the H1 collapse is eliminated. Raw
OSC equals the decomposition success rate, so the bottleneck is **header-path
decomposition**, localized to the row axis. (E2)

**5.3 Row axis — total-row augmentation.** Diagnosis: **68%** of row-axis misses are
share/ratio queries needing an unnamed total row. Treatment lifts **row-cov 0.615 →
0.888** and OSC 0.416 → 0.652 (paired ΔOSC **+0.236**, CI [0.174, 0.304]). (diag, E6)

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

**Honest position.** Enumeration does **not** beat dense top-k on *raw* OSC (0.65 <
0.79); the contribution is **scope-robustness + a completeness guarantee + diagnosis-
driven axis fixes + the schema-linking column result**, not a higher OSC number.

### Open / limitations
- Column completeness on two-entity comparisons (~25% residual) — needs heavier
  methods (LLM/fine-tuned schema linker); future work.
- End-to-end answer accuracy is solver-limited (8b); a stronger solver is needed.
- Single dataset (HiTab); selection/comparison aggregations excluded.
