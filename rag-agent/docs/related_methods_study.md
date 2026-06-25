# How other papers select cells / sub-tables (literature study)

Goal: understand how prior work retrieves the cells needed to answer table questions,
and whether any targets **operand-set completeness for aggregation on hierarchical
tables** (our Problem A). Method-level notes; confidence flagged per row.

## Per-paper method notes

### TableRAG (2410.04739, Google, NeurIPS'24) — cell-level table RAG
- **How it selects:** query expansion → **schema retrieval** (which columns) +
  **cell retrieval** (which cell values), then feeds only those to the LLM.
- **Objective:** efficiency + precision (shorten prompts, cut info loss) — **not a
  completeness guarantee**.
- **Tables:** flat, *million-token* large tables (Arcade, BIRD-SQL).
- **Aggregation/hierarchy:** not the focus; no operand-set completeness.
- *Closest cousin* to us (cell-level retrieval) but flat + precision-driven, no 100%.

### H-STAR (2407.05952) — hybrid SQL+semantic sub-table extraction
- **How it selects:** two steps — **columns then rows**. Each step = SQL extraction
  **∪** text/semantic verification ("multi-view"; the text pass *recovers rows SQL
  missed*). Reduces processed cells **159 → 18** on WikiTQ.
- **Objective:** answer accuracy, *not* guaranteed completeness (their error analysis:
  extraction causes only 11–23% of failures).
- **Tables:** flat Wikipedia (WikiTQ 68.85%, TabFact 83.74% w/ GPT-3.5).
- **Aggregation/hierarchy:** not addressed.
- *Lesson to borrow:* the **col→row split** and the **"verification pass to catch
  missed rows"** mirror our row/col axes + total-row augmentation. Our hybrid
  (embed rows / lexical cols) is the same spirit as their SQL∪semantic union.

### Chain-of-Table (2401.04398) — iterative table transformation
- **How it selects:** doesn't retrieve a subset; the LLM **transforms the whole
  table** step by step (add column, select row, group…) as a reasoning chain.
- **Objective:** answer accuracy; no evidence-completeness notion.
- **Tables:** flat (WikiTQ, FeTaQA, TabFact). Not built for large/hierarchical.

### TabSieve (2602.11700) — in-table evidence selection
- For **tabular *prediction*** (TabNet/TabPFN setting), not QA aggregation; selects
  evidence cells for accuracy, **no completeness guarantee**. Different task.

### HiTab original (2108.06712) — hierarchy-aware semantic parser
- **How it selects:** trains a MAPO + TaBERT parser to produce a logical form;
  uses `linked_cells` as *partial supervision*, not as a retrieval target.
- **Objective:** end-to-end denotation accuracy (**45.5% dev**), not minimal complete
  retrieval. Same dataset as us; BERT-era, no generative LLM.

### (from `RELATED_DELTA.md`, method-verified earlier)
- **DCTR / Huawei-TableRAG / TableRAG-SQL:** retrieve schema+values, **defer
  aggregation to SQL** on flat relational schemas — aggregation removed from
  retrieval.
- **MixRAG / HD-RAG (2504.09554):** models an internal header tree but only to pick
  the **top-1 document**; never enumerates the scope's cells. Benchmark = DocRAGLib,
  metric = top-1 retrieval (not cell-level).
- **Graph-Table-RAG / T-RAG (2504.01346):** corpus-level *table* selection; no
  operand labels.

## Cross-cutting findings (what they all share = our gap)

1. **Objective is answer-accuracy or efficiency — never all-or-nothing completeness.**
   No prior method treats "retrieve *every* operand, miss none" (OSC) as the goal.
2. **Almost all are flat tables** (Wikipedia / relational). Hierarchical (HiTab) is
   acknowledged as *harder* but under-served for cell selection.
3. **Selection is by similarity / SQL / LLM** — not by **header-tree structure
   enumeration**.
4. **No completeness guarantee.** They report graceful recall; none guarantee 100%,
   and none handle the *unnamed total/denominator row* (our diagnosed 68% failure).
5. **The 2025 survey (2510.09671) states it outright:** operand-set completeness for
   aggregation and formal completeness guarantees are *not addressed*, and there is
   *no standard precision/recall evaluation of evidence-cell selection* — "an
   underexplored area." ← strongest external validation of our gap.

## Where this leaves Problem A

The intersection **{hierarchical headers} × {operand *set*, all-or-nothing
completeness} × {completeness guaranteed, minimize cells}** is unoccupied. The
nearest works (TableRAG, H-STAR) optimize precision/accuracy on *flat* tables
without a completeness guarantee; the hierarchical works (HiTab parser, HD-RAG)
don't frame retrieval as minimal-complete operand selection.

**Borrowable ideas:** H-STAR's col→row two-step + verification pass (catch missed
rows) and TableRAG's query expansion (could surface the unnamed total row) are worth
adapting as completeness-oriented baselines/components.

> Confidence: method-verified for H-STAR, the survey, TabSieve, and the four in
> `RELATED_DELTA.md`; TableRAG and Chain-of-Table from abstract + survey (read full
> method sections before final citation).
