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
