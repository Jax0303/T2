# Citation verification log (against primary sources)

Per the honesty guard (auto-summaries hallucinate), every citation carrying a
specific claim in PAPER_DRAFT.md was checked against the primary source.
Date: 2026-07-03. ✅ = verified verbatim; ⚠️ = needs a wording fix before submission.

## ✅ Verified

- **HiTab** — Cheng et al., *HiTab: A Hierarchical Table Dataset for Question
  Answering and Natural Language Generation*, ACL 2022. arXiv:2108.06712,
  aclanthology 2022.acl-long.78. Abstract confirms: hierarchical tables from
  statistical reports + Wikipedia, 10,686 QA pairs / 3,597 tables, entity &
  quantity alignment annotations (= our `linked_cells` source).
- **MAPO w/ hierarchy-aware logical form, 45.5% dev** — ⚠️ *number verified but
  source needs precision*: 45.5% dev / 42.3% test is from the **official repo
  README released checkpoint** ("slightly higher than the results in paper due to
  the updated dataset"), not the paper table. Cite as "official released
  checkpoint on the updated dataset (repo README)".
- **OHD** — Cao et al., *Orthogonal Hierarchical Decomposition for
  Structure-Aware Table Understanding with Large Language Models*,
  arXiv:2602.01969. Abstract confirms our §5.9 description: Orthogonal Tree
  Induction (column tree + row tree), dual-pathway cell lineage, LLM semantic
  arbitrator, evaluated on AITQA + HiTab, **representation for LLM input — no
  retrieval/selection**. Our `ohd_lite` omissions (learned tree induction,
  arbitrator) are accurately disclosed.
- **FT-RAG** — Guo, Geng, Mao, *FT-RAG: A Fine-grained Retrieval-Augmented
  Generation Framework for Complex Table Reasoning*, arXiv:2605.01495 (May 2026).
  Metrics: table-level & cell-level **Hit Rate**, exact-value accuracy recall —
  graceful/partial metrics, **not all-or-nothing completeness** → our gate claim
  holds.
- **Topo-RAG** — arXiv:2601.10215 (Jan 2026), *Topology-aware retrieval for
  hybrid text-table documents*. Cell-aware late interaction (WARP), metric:
  **nDCG@10** on SEC-25 — ranking metric, not completeness → gate claim holds.
- **2025 TQA survey** — arXiv:2510.09671, *Table Question Answering in the Era
  of Large Language Models: A Comprehensive Survey of Tasks, Methods, and
  Evaluation* (Oct 2025). Exists; our stronger paraphrase "confirms OSC is
  unmeasured" should be softened to "does not list an evidence-completeness
  metric" unless a direct quote is found (re-check §evaluation of the survey
  when writing the camera-ready related work).
- **RESDSQL** — Li et al., AAAI 2023: cross-encoder (RoBERTa) ranking-enhanced
  schema linking. Real, correctly described.
- **MT2Net retriever recall** (verified 2026-07-15, ar5iv full text of
  arXiv:2206.01347): "76.4% recall for the top-10 retrieved facts and 80.8%
  recall for the top-15" — *partial* recall of supporting facts, retrieved
  **within a single document** by a BERT bi-classifier over cell-sentences
  ("For Innovation Systems of Segment, sales of product in 2018, ... is
  2,894"). Note for related work: MultiHiertt itself already serializes each
  cell into a sentence with its hierarchical row+column headers — cite when
  positioning S3-style serialization (the primitive is theirs; our delta is
  the corpus-level retrieval objective + OSC).
- **HotpotQA supporting-fact EM** (verified 2026-07-15, ar5iv 1809.09600):
  set-level exact match confirmed ("Joint EM is 1 only if both tasks achieve
  an exact match and otherwise 0"); baseline Sup-EM 21.95 (distractor) /
  5.28 (fullwiki) vs Sup-F1 66.66/40.98 — the EM-vs-F1 gap is the canonical
  precedent for "all-or-nothing is much harsher than partial credit".

## ⚠️ Fix before submission

- **"CE-SL"** is NOT a standalone paper — it is a baseline label used in later
  work (e.g. AutoLink, arXiv:2511.17190: "cross-encoder (BGE-reranker) scores
  query × column"). Replace with: RESDSQL (AAAI 2023) + *Extractive Schema
  Linking for Text-to-SQL* (arXiv:2501.17174) and/or cite AutoLink for the
  CE-SL terminology.
- **MAPO 45.5%** — cite the repo README checkpoint, not the paper (see above).
- **Survey "confirms unmeasured"** — soften or find the exact quote.

## Still to verify when writing related work

- TableRAG (two distinct papers exist: NeurIPS 2024 million-token + arXiv
  2506.10380 heterogeneous-document — disambiguate which we mean).
- H-STAR, Chain-of-Table, HD-RAG, AITQA (Katsis et al.) — standard, but pull
  exact venues/years for the .bib.
- MultiHiertt — Zhao et al., ACL 2022 (add: it is our §5.8 second OSC dataset).
