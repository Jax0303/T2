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

- **Header/data row classification (table structure recognition)** — supports the
  numeric-vs-text header-boundary heuristic `guess_n_header_rows` (§3.0). Framing:
  cite the *principle* as standard practice; the ≥50%-numeric threshold + 4-digit-year
  exclusion is our deterministic instance, **not** any single paper's rule verbatim
  (corrects an earlier misstatement that the rule was borrowed as-is).
  - **Adelfio & Samet**, *Schema Extraction for Tabular Data on the Web*,
    PVLDB 6(6):421–432, 2013. ✅ verified from primary PDF (vldb.org/pvldb/vol6/p421):
    CRF row classification into header(H)/data(D)/metadata/aggregate; feature list
    includes `IsNumeric?`; verbatim "Header rows often contain relatively short
    textual values, rather than numbers or dates" — directly grounds the heuristic.
  - **Cafarella et al.**, *WebTables: Exploring the Power of Tables on the Web*,
    PVLDB 1(1):538–549, 2008. Foundational web-table header detection. ⚠️ re-verify
    full text for the header-detection mechanism before camera-ready.
  - **Fang, Mitra, Tang, Giles**, *Table Header Detection and Classification*,
    AAAI 2012. Title/venue via search (cdn.aaai.org/ojs/8206); ⚠️ verify primary.
  - **Zhang & Balog**, *Web Table Extraction, Retrieval and Augmentation: A Survey*,
    2020, arXiv:2002.00207. Survey anchor — header detection is an established subtask.

## ⚠️ Fix before submission

- **"CE-SL"** is NOT a standalone paper — it is a baseline label used in later
  work (e.g. AutoLink, arXiv:2511.17190: "cross-encoder (BGE-reranker) scores
  query × column"). Replace with: RESDSQL (AAAI 2023) + *Extractive Schema
  Linking for Text-to-SQL* (arXiv:2501.17174) and/or cite AutoLink for the
  CE-SL terminology.
- **MAPO 45.5%** — cite the repo README checkpoint, not the paper (see above).
- **Survey "confirms unmeasured"** — soften or find the exact quote.

## Venues for the .bib (verified 2026-07-15, primary sources)

- **MultiHiertt** — Zhao et al., *ACL 2022* (2022.acl-long.454, pp. 6588–6600,
  Dublin). Our §5.8/§5.1c second OSC dataset.
- **AIT-QA** — Katsis et al., *NAACL 2022 Industry Track*
  (2022.naacl-industry.34, pp. 305–314; arXiv 2106.12944).
- **Chain-of-Table** — Wang et al., *ICLR 2024*.
- **H-STAR** — Abhyankar, Gupta, Roth, Reddy, *NAACL 2025*
  (arXiv 2407.05952; official repo confirms NAACL 2025).
- **HD-RAG / MixRAG** — arXiv 2504.09554, **no published venue found** as of
  2026-07; note the paper was RENAMED on arXiv: v1 "HD-RAG: Retrieval-Augmented
  Generation for Hybrid Documents..." → latest "Mixture-of-RAG: Integrating
  Text and Tables with Large Language Models". Cite as arXiv preprint under the
  latest title, mention the HD-RAG alias (FT-RAG cites it as MixRAG).
- **TableRAG disambiguation** — two distinct papers: (a) Chen et al., NeurIPS
  2024, million-token table understanding; (b) arXiv 2506.10380 (Huawei),
  heterogeneous-document RAG with SQL execution. WE mean (b) in RELATED_DELTA
  ("Huawei-TableRAG"); if (a) is ever cited, use "TableRAG (NeurIPS 2024)"
  explicitly.
