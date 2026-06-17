# F2+F4 — Structure-preserving serialization protects numeric computation in table RAG

## Thesis (one sentence)

> In retrieval-augmented generation over tables, chunking/flattening a large or
> hierarchical table into text loses header structure (**F2**), and that loss
> breaks the downstream **numeric computation** (**F4**); a structure-preserving
> serialization that re-attaches the full hierarchical header path to every cell
> restores numeric-answer accuracy — and we measure this on **end-to-end answer
> correctness**, not retrieval recall, which is the loop prior work leaves open.

## The gap (verified, 2024–2026)

Two independent adversarial literature sweeps converged on the same open slot:

| System | Structure-preserving chunk/repr? | Measured on **numeric computation**? |
|---|---|---|
| STC — *Structure-Aware Chunking for Tabular RAG* (arXiv:2605.00318) | ✅ Row-Tree, headers ride in row chunks | ❌ retrieval only (MRR/Recall@1; MAUD legal, non-numeric) |
| GTR / Graph-Table-RAG (arXiv:2504.01346) | ✅ anti-"vector-dilution" graph index | ❌ QA acc reported, numeric not isolated |
| TableRAG (NeurIPS 2024, arXiv:2410.04739) | ❌ schema + distinct-cell retrieval | ⏭️ sidesteps — offloads arithmetic to code execution |
| TableGPT2 (arXiv:2411.02059) | ❌ | ⏭️ sidesteps — code sandbox |
| HD-RAG (arXiv:2504.09554) | ✅ row-and-column representation | ~ improves Hit@K / EM (closest prior; not numeric-isolated) |
| T²-RAGBench (EACL 2026, arXiv:2506.12071) | — (benchmark) | ✅ shows best RAG pipeline still **~30% below oracle** on Number-Match → problem **unsolved** |

**Empty cell = our contribution:** a chunking/serialization scheme *measured on
end-to-end numeric-answer accuracy* over hierarchical tables. Supporting evidence
that the mechanism is real but under-evaluated at the RAG-pipeline level:
"Same Content, Different Representations" (ICLR 2026, arXiv:2509.22983) — NL2SQL
drops up to **−45 pp** on semi-structured input with content held constant;
HiTab API-assisted multi-index vs flattening **+10.3%** (arXiv:2310.14687); OHD
hierarchical decomposition **+20 pts** over flattening (arXiv:2602.01969).

## Operationalization

Retrieval is held fixed (the answer cells are present), so the **answer stage is
isolated** — the only thing that varies is how much table **structure** survives
into the text the model reads. Three serialization levels (independent variable):

- **S0 `flat_values`** — data values only, no headers. Models a mid-table chunk
  whose header row was lost at the chunk boundary.
- **S1 `flat_leaf`** — leaf column header + leaf row label. Models a header-aware
  chunker that nonetheless **flattens the hierarchy** (drops parent headers).
- **S2 `header_path`** — full hierarchical header path on every cell/column.
  Structure-preserving.

The LLM reads the serialized excerpt and computes the answer directly (this is how
RAG QA actually works: retrieved text → answer). Metric = `numeric_match | exact_match`.
No gold answer is ever shown to the model.

### Stratifier — table type (the causal control)

- **flat** (WikiSQL, exact SQL-derived gold): no hierarchy to lose → S0<S1≈S2.
- **hier** (HiTab, hierarchical headers): S2 should beat S1 should beat S0, and the
  **S2−S1 gap grows with hierarchy depth**.

If the gain appeared on flat tables too, it would be a verbosity artifact, not
structure. The flat split is the falsification control: the effect must be
**caused by hierarchy preservation**, concentrated on hier.

### Worked illustration (real HiTab table, dev)

Under `flat_leaf` the column headers collapse to `percent | from | to | percent | from | to`
— the **women vs men** distinction (a hierarchy parent) is destroyed, so "which
group" / per-group arithmetic is unanswerable. `header_path` keeps
`women > percent` vs `men > percent`. This is the F2→F4 failure in one cell.

## Harness

`scripts/codegen_chunk_eval.py` — paired across conditions (same question),
paired bootstrap 95% CI on per-condition deltas, seed=42, Groq LLM.
`--max-rows` is the chunk budget (rows serialized).

```
python scripts/codegen_chunk_eval.py --n 40 --splits flat,hier \
    --llm groq:llama-3.3-70b-versatile --out results/codegen/chunk_struct.json
```

## Planned ablations / next

- [ ] **Chunk-budget sweep** (`--max-rows`): accuracy vs degree-of-truncation curve,
      S0/S1/S2 — the "accuracy vs structure-loss" plot no prior paper reports.
- [ ] **Complexity stratification** within hier (single-cell lookup → multi-op
      formula) to show the S2−S1 gap scales with reasoning depth.
- [ ] **Model control** (≥2 LLMs) so the delta is not model-specific.
- [ ] **PoT variant**: code-execution answer over the serialized text, to confirm
      the structure effect survives when arithmetic is offloaded to code.
- [ ] Position vs HD-RAG / STC: we report **numeric-answer accuracy**, they report recall.

## Honest caveats

1. Contribution is partly **confirmatory consolidation** (mechanism known); strength
   comes from closing the measurement loop + the flat-vs-hier causal control.
2. F4 reintroduces **LLM dependence** → report model-controlled deltas, watch Groq limits.
3. Competitors are fast-moving (all 2025–26) → differentiate sharply on "numeric
   computation under chunking," not recall.
