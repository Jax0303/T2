# Paper outline (working)

**Working title:** *Operand-Set Completeness: Header-Tree Retrieval for Aggregation
over Hierarchical Tables*

**One-sentence thesis:** Aggregation over hierarchical tables requires retrieving
*every* operand cell (all-or-nothing), which similarity retrieval cannot guarantee;
representing the header as a tree and enumerating a query-resolved scope node makes
the operand set complete-by-construction and re-localizes the problem to header-path
decomposition, which we then diagnose and treat axis-by-axis.

---

## 1. Introduction
- Hierarchical tables (stats reports, finance) + aggregation questions are everyday.
- A correct aggregation answer needs **all** operand cells — a *completeness*
  requirement, unlike single-passage QA where the top hit suffices.
- Similarity RAG ranks by *relevance*, not *completeness* → it drops operands as the
  aggregation scope grows.
- **Contributions:** (i) formalize **Operand-Set Completeness (OSC)** as a
  retrieval objective + metric; (ii) **header-tree enumeration** — complete-by-
  construction, scope-robust; (iii) a **diagnosis** of the residual bottleneck
  (row: unnamed *total* rows; col: *metric* columns); (iv) **targeted treatments**
  (total-row augmentation; cross-encoder column selection); (v) an honest
  **completeness↔precision frontier** ("100% in a small set" is open).

## 2. Related work (delta in `RELATED_DELTA.md`, `related_methods_study.md`)
- **Table QA / parsers** (HiTab MAPO): end-to-end answer accuracy, not minimal-
  complete retrieval.
- **Table-RAG / cell selection** (TableRAG, H-STAR, Chain-of-Table): precision /
  token-efficiency on **flat** tables; **no completeness guarantee**.
- **Hierarchical representation** (OHD 2602.01969, HD-RAG): same orthogonal row/col
  trees but to **serialize the whole table** for accuracy — **no retrieval/selection**.
- **Schema linking** (CE-SL, RESDSQL): column selection via **cross-encoders** (we
  adopt this for our column axis).
- **Gap (a 2025 survey confirms):** operand-set completeness for aggregation and
  evidence-cell precision/recall are *underexplored*. No prior work makes header-tree
  scope enumeration / OSC a retrieval-time objective.

## 3. Problem formulation & metric
- Query *q*, hierarchical table *T* (top/left header trees), gold operand set *G*.
- **OSC(q) = 1 iff G ⊆ retrieved** (all-or-nothing). Necessary condition for a correct
  aggregation; strictly harder than averaged cell recall.
- Population: HiTab dev arithmetic aggregations, scope **m≥2** (n=161); selection/
  comparison excluded (value-matching can't build gold). `eval/operand_set.py`.

## 4. Method
- **4.1 Header-tree scope enumeration** — a header node *is* an aggregation scope;
  enumerate its numeric leaves → operand set **complete-by-construction**
  (`header_enum.py`). OSC | decomposition-correct = 1.0, flat in m.
- **4.2 Query→node decomposition** — embedding **tree-node** resolver for rows
  (semantic, closes vocabulary/depth); **cross-encoder** for columns (schema-linking
  SOTA) (`header_embed_resolver.py`).
- **4.3 Diagnosis-driven augmentations** — `total_like_rows` (ratio queries need the
  unnamed denominator), `expand_sibling_groups`.
- **4.4 Recall-first guarantee** — union/fallback to enforce 100% completeness when
  required (the professor's constraint); minimize the set under it.

## 5. Experiments (all paired, HiTab dev m≥2)
- **5.1 (H1) Similarity OSC collapses with scope** (E1): m=2→0.60, m=9+→0.29.
- **5.2 (H2) Enumeration re-localizes the bottleneck** (E2): OSC|decomp=1.0 flat;
  raw OSC = decomposition success rate; bottleneck = row-axis decomposition.
- **5.3 Depth is a method-specific liability, causally** (E3); embedding resolver
  closes it (matches a 70b LLM, LLM-free).
- **5.4 Row axis — diagnosis + treatment** (diag_row_failures, E6): 68% = total-row
  pairing; total-row augmentation → **row-cov 0.615→0.888**, ΔOSC +0.236
  [0.174,0.304].
- **5.5 Column axis — diagnosis + cross-encoder** (retrieval_algorithms_study, E10):
  74% of unpinned cols need 1 *metric* column; cross-encoder ranks "%"↔"percentage",
  removes whole-axis dumps (cells 40→31) — precision win; **completeness open**
  (whole-axis fallback is trivially complete).
- **5.6 Completeness↔precision frontier** (E5): precise 0.42@19c … complete 1.0@160c;
  recall-first union = 1.0@123c. "100% in a small set" open.
- **5.7 Generation stage (supporting)** (E4, E7): structured (header-path=value)
  context cuts silent errors (+0.24, oracle-fixed); end-to-end, a weak 8b solver
  floors all retrieval arms (oracle 0.61 vs ~0.13) → **precision dominates**, and
  retrieval gains need a stronger solver to surface.

## 6. Discussion / limitations (honest)
- Enumeration does **not** beat dense top-k on *raw* OSC (budget); the edge is
  **scope-robustness + completeness guarantee + diagnosis-driven fixes**, not a
  higher OSC number.
- **Column completeness** (exact column on unnamed/metric axes) is the top open
  problem; cross-encoder improves precision, not coverage.
- End-to-end answer accuracy is **solver-limited** (8b); needs a stronger model.
- Single dataset (HiTab); selection/comparison excluded; do **not** claim the
  orthogonal-tree representation or the (header-path=value) format as novel (OHD).

## Contribution scorecard (what's solid vs open)
| claim | status |
|---|---|
| OSC formalized as a retrieval objective | **solid** (novel; survey gap) |
| Enumeration is scope-robust (OSC\|decomp=1.0) | **solid** (E2) |
| Embedding resolver closes row vocab/depth (=70b, LLM-free) | **solid** |
| Total-row augmentation fixes the row axis (+0.27 row-cov) | **solid** (E6) |
| Column axis is schema-linking; cross-encoder best column selector at budget (col-recall@2 0.40→0.61) | **solid at-budget** (full completeness still open) |
| Beats dense on raw OSC | **no** (honest) |
| End-to-end answer gain from better retrieval | **unproven** (solver-limited) |
