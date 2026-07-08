# W0 — Differentiation gate (§6 of the research spec)

Gate question: does any of the four nearest works **put aggregation-scope
enumeration on the table's header tree as a retrieval-time objective that
guarantees operand-set completeness**? Verified against the method sections
(arXiv HTML, fetched 2026-06). Conclusion: **gate PASSES — the axis is empty.**

| Work | What retrieval optimizes | Where aggregation happens | Hierarchical headers? | Operand-set completeness as retrieval objective? |
|---|---|---|---|---|
| **DCTR** (2603.07146) | schema + value components, top-k tables (capped recall) | downstream text-to-SQL | flat relational only | **No** — "In the retrieval phase, only schema and value components are used while the aggregator components benefit the downstream tasks, e.g. text-to-SQL." |
| **Huawei-TableRAG** (2506.10380) | top-N cosine-similar chunks | **SQL execution after retrieval** | flat relational only (`table_name, columns[ColName,Type,Examples]`) | **No** — pipeline is "query decomposition → text retrieval → SQL programming and execution"; mitigation is post-hoc cross-validation, not a retrieval constraint |
| **HD-RAG / MixRAG** (2504.09554) | most-relevant **document D\*** (top-1) | **RECAP** post-retrieval prompting + external calculator | yes, internal H-RCL header tree (for representation/summarization) | **No** — retrieval is document-level + relevance filtering; completeness of the operand cell set is never an objective |
| **T-RAG / Graph-Table-RAG** (2504.01346) | corpus-level coarse-to-fine **table selection** | downstream LLM reasoning | "hierarchical" = corpus index, not table header tree; relational schema assumed | **No** — MultiTableQA has **no aggregation labels / linked operand cells** |

## Precise deltas (so the contribution is not "decompose + SQL")

1. **vs DCTR / Huawei-TableRAG.** Both explicitly *remove* aggregation from
   retrieval and defer it to SQL on a **flat relational schema**. Our setting has
   no relational schema — the header is a tree whose nodes *are* aggregation
   scopes. We make the scope the retrieval-time objective instead of deferring it.

2. **vs HD-RAG.** Closest threat: it *does* model the internal header tree
   (H-RCL). But it uses the tree to **represent/summarize** a table for **top-1
   document** retrieval, then computes via RECAP reasoning. It never **enumerates
   the leaves under a scope node at retrieval time** to guarantee the operand set
   is complete. Our delta is exactly that enumeration + the OSC objective.

3. **vs T-RAG.** "Hierarchical" is a corpus index for *selecting* tables, not the
   intra-table header tree. Its benchmark lacks operand-cell labels, which is also
   why we evaluate on **HiTab** (linked_cells/answer_formulas give gold operands).

## Residual honesty
- HD-RAG genuinely handles hierarchical *internal* structure, so the delta must be
  stated at the **retrieval-objective** level (completeness-by-enumeration), not as
  "nobody models header trees."
- All completeness claims here are about the *retrieval objective*; none of the
  four formalize OSC or measure it. Our contribution (i) OSC formalization,
  (ii) deterministic header-tree enumeration, (iii) completeness-as-SOTA-limit
  diagnosis remains distinct.

**Gate verdict: proceed.** No re-design needed; §3 hypotheses stand.

---

## OHD — Orthogonal Hierarchical Decomposition (2602.01969) — closest *representation* work

Verified against the full method + experiments (arXiv HTML, 2026; marked "work in
process"). This is the **nearest work on the representation side** and the most
important to differentiate: it decomposes a hierarchical table into **orthogonal
row-tree + column-tree** on **HiTab** — the same structural primitive we use.

**What OHD does (method):** Orthogonal Tree Induction (build column tree 𝒯_col and
row tree 𝒯_row from geometry + LLM semantic predicates) → Dual-Pathway Association
(linearize each tree, cross-tree-supplement each cell) → an **LLM "semantic
arbitrator"** picks the best of the column-major / row-major **serializations of the
ENTIRE table**. Each cell is rendered `prelineage ⊕ (orthogonal-header ⇒ value)`
("Context → Key → Value"). Metric: **end-to-end answer accuracy only** (EM + LLM-eval;
HiTab 60.07 EM full / 64.74 EM on a 50×50 subset, Qwen2-72b). **No** cell selection,
**no** recall/precision, **no** operand-set completeness, **no** aggregation/total
handling.

### Differentiation table (OHD vs ours)

| axis | OHD (2602.01969) | ours |
|---|---|---|
| **primary goal** | structure-aware **representation** → end-to-end accuracy | **retrieval**: minimal **complete** operand set |
| **what the orthogonal trees are used for** | **serialize the *whole* table** for the LLM to read | **enumerate the aggregation scope** to *select* cells |
| **cell selection / retrieval** | **none** — feeds the entire table | **yes** — retrieves the operand subset |
| **objective metric** | EM / LLM-eval accuracy | **OSC** (all-or-nothing completeness) + cells + answer acc |
| **aggregation operand set** | not distinguished from lookup | central; **total/denominator-row failure diagnosed (68%)** |
| **large-table scalability** | limited (whole-table serialization → context blows up) | retrieval keeps the context minimal |
| **per-cell lineage repr.** | yes, over the whole table (its core) | yes, but only as the E4 *generation format* on already-retrieved cells |
| **dataset** | HiTab, AITQA | HiTab |

### Precise deltas / residual honesty
1. **Shared primitive, opposite use.** Both build orthogonal row/column header trees.
   OHD uses them to *represent the whole table*; we use them to *enumerate a scope and
   retrieve the minimal complete cell set*. Literally orthogonal contributions
   (representation vs retrieval) — they are **complementary, not competing** (OHD's
   serialization could be our context formatter; our retrieval could shrink OHD's
   whole-table input).
2. **Do NOT claim the orthogonal-tree representation as our novelty** — OHD has it.
   Our novelty is the **retrieval objective**: OSC formalization, completeness-by-
   enumeration, the total-row diagnosis + treatment, and minimal-set-under-100%.
3. **E4 caveat:** OHD's `Context→Key→Value` per-cell representation overlaps our E4
   structured `(header-path = value)` format — so **do not claim E4's format as novel
   either**; cite OHD and frame E4 only as a controlled *format-effect* measurement on
   fixed retrieval.
4. OHD is the natural **"whole table, well-represented" baseline** our argument targets:
   it never selects, so it inherits the context-length / scalability cost we avoid by
   retrieval. A fair head-to-head = OHD-serialization-of-whole-table vs ours at matched
   accuracy, reporting context size.

**Gate still passes** on the *retrieval-completeness* axis, but OHD tightens the
representation axis: the paper must be cited, the tree representation must **not** be
claimed as novel, and the delta stated at the objective level (retrieval/OSC).

---

## W1 — Freshness re-check (2026-07-08, web sweep for post-gate publications)

Question: did anything published since the W0 gate (fetched 2026-06) put
operand-set completeness on the table as a retrieval-time objective?
**Answer: no — gate still passes.** Two new works must be cited; one honest
tightening of the OSC-novelty claim is required.

### New works found (cite both; neither competes on our axis)

| Work | What it does | Why it does not scoop us |
|---|---|---|
| **Topo-RAG** (2601.10215, 2026-01) | Hybrid text–table retrieval; tables scored by **Cell-Aware Late Interaction** (ColBERT-style MaxSim between each query token and each cell vector) | Still **relevance ranking**: MaxSim has no term that lifts a cell no query token resembles. An unnamed total row (header empty/"total") matches no query token, so it stays unreached **by construction** — i.e. Topo-RAG sits *inside* the paradigm §5.1b diagnoses. Strengthens our claim ("even cell-level late interaction shares the ceiling"); candidate future baseline. No completeness metric, no header-tree scopes, no aggregation handling. |
| **ASTRA** (2604.08999, 2026-04) | LLM reconstructs the table into a Logical Semantic Tree (AdaSTR), then dual-mode reasoning: tree-search navigation + code execution (DuTR). Evaluated on **HiTab and AIT-QA** | Same family as OHD: whole-table **representation + reasoning**, no retrieval/cell selection, no recall or completeness measurement, no total-row handling. Must be cited (shares our datasets); slots into the E9 frame as another "serialize/reason over the whole table" counterpart, inheriting the same context-scaling cost. |

### Honest tightening — the all-or-nothing *idea* has cousins in multi-hop text QA

**Perfect Recall@K** (PR@K: 1 iff *all* relevant objects are in the top-K; e.g.
PRISM, arXiv 2510.14278) and "complete evidence set" scoring (FEVEROUS-style;
multi-hop evidence pursuit) already exist for **passage/sentence evidence in
multi-hop text QA**. Therefore:

- Do **NOT** claim we invented all-or-nothing retrieval evaluation.
- Claim precisely: **OSC instantiates the all-or-nothing objective for the
  operand-cell sets of aggregation over hierarchical tables**, where the 2025
  TQA survey confirms it is unmeasured — and the paper's weight rests on what
  the metric *reveals* (the §5.1b structural ceiling: 28.5% unnamed-total
  operands, median rank 39.5 vs 8, 76% of dense incompletes) and on the
  structural completeness guarantee, not on the metric alone.
- Action: one-sentence citation of the PR@K / complete-evidence-set lineage in
  Related Work, positioned as "all-or-nothing objectives exist for text
  evidence; tables' operand sets differ because the missing member is
  *structurally* unreachable by similarity, not merely under-ranked."

**W1 verdict: proceed.** Contribution ordering (diagnosis = headline, OSC =
contribution 1) already absorbs the PR@K lineage without reframing.
