# RAG over hierarchical-header tables

Masters-thesis research code. The working project is **[`rag-agent/`](rag-agent/)**.

Read in this order:

1. [`rag-agent/CLAUDE.md`](rag-agent/CLAUDE.md) — what the thesis is (2026-08-30 redefinition),
   the current priorities, the rejected hypotheses, and the citation rules. **Authoritative.**
2. [`rag-agent/RESULTS.md`](rag-agent/RESULTS.md) — every current number with the results file
   it came from. Pre-2026-08-31 numbers live in
   [`rag-agent/RESULTS_ARCHIVE-2026-08-30.md`](rag-agent/RESULTS_ARCHIVE-2026-08-30.md).
3. The latest `rag-agent/HANDOFF-<date>.md` — where the last session stopped.
4. [`rag-agent/PRESENTATION-2026-09-04.md`](rag-agent/PRESENTATION-2026-09-04.md) — the talk,
   one chapter per experiment, each line sourced.

**Rule this repo runs on:** *every number carries the results file it came from.*
A number without a source does not go in a document.

---

## The line in one paragraph

Hierarchical-header tables cannot go into a RAG prompt whole (86% of RealHiTBench gold
tables exceed 512 tokens) and cannot be flattened without losing what a value means.
So one data cell becomes one sentence — *table title + row header path + column header
path + value* (`S3c`) — the whole corpus of cell sentences is indexed once, and a query
is answered from the top-k cells a hybrid BM25 + dense index returns. The encoder is
fine-tuned offline once (`models/bge-base-cell-ft-p0`); per-query cost stays one ANN
lookup. The reader is a local Qwen2.5-7B (4-bit).

Stage 1 (retrieval) is scored as *query type × (table hit, cell hit given the table)* —
`rag-agent/results/stage1/BOARD.md`. Stage 3 (answer EM) is scored on the same queries
with the same cells injected — `rag-agent/results/stage3/`.

## Positioning — what is prior art (do not present as novel)

- **"Verbalize each cell with its hierarchical row/column headers, retrieve top-n"** is
  MT2Net (Zhao et al., *MultiHiertt*, ACL 2022, arXiv:2206.01347 §4), verbatim. Adopt and
  cite. The difference from MT2Net is **per-query cost and scope**: MT2Net scores every
  candidate with a cross-encoder inside one document; this indexes the corpus once and
  does one ANN lookup (`rag-agent/CLAUDE.md` §2).
- The header-path-vs-flat serialization gain is also published: OHD (arXiv:2602.01969)
  Table 2, and MT2Net's own flat-vs-hierarchical comparison.
- **"Fix MT2Net's 64% fact-integration error"** misreads MT2Net Table 6 — that table
  classifies the generated program, not retrieval failures.
- **Re-retrieval to recover missing operands** was measured here and is a budget change
  in disguise (+.014, p=.55 against a per-query budget-matched control). Any arm that
  grows the evidence set must report such a control before its delta is quoted.

## Archived measurements (2026-08, OSC line)

The repository's first line scored *Operand-Set Completeness* (all gold operand cells
retrieved, or nothing) and built operand-targeted retrieval, structural injection and
a completeness gate on top of it. Those interventions were measured against budget-matched
controls and **lost**; the line was closed on 2026-08-30 and its code was removed from
the tree on 2026-09-05 (history before commit `b2fddb8`; result files at `753fa2e`).
Two measurements from it are still cited in the talk and are kept in
`RESULTS_ARCHIVE-2026-08-30.md` §D and §I:

- RealHiTBench, raw hierarchical tables, same retriever and k, only the serialization
  differs: S1 flat → S2 header-path lifts strict EM on gpt-5.1 .160 → .309 (n=94,
  p=.0043), gpt-4.1-mini .170 → .277 (p=.0213), llama-3.3-70b .205 → .341 (n=44, p=.0703).
- Six shuffle draws of a length-matched control: the header-path gain comes from the
  ancestor header **words**; the hierarchical **order** effect is not detectable
  (hybrid S2 sits inside the shuffle distribution, best permutation p ≈ .14).

`rag-agent/RESEARCH_STRUCTURE.md` is that line's design document, kept for its
related-work list and IP notes; where it conflicts with `CLAUDE.md`, `CLAUDE.md` wins.
