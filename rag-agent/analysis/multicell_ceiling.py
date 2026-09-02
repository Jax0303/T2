#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reader ceiling on the multi-cell lookups: gold cells injected, no retrieval.

HANDOFF 2026-09-01 makes this the first GPU job, because it decides whether the
retrieval axis is worth any more work on this task. Phase 4 measured the
single-cell ceiling at .9206 on ``hitab_lookup``; a multi-cell answer needs
EVERY cell right for EM=1, so the arithmetic prediction from the dev cell-count
distribution (2 cells x23 / 3 x5 / 4 x4 / 5 x1) was ~.82 -- a calculation, never
measured. If the measured number lands below .9, no amount of retrieval fixes
multi-cell answering and the lever is the reader.

Two injection sets per query, because the population and the metric disagree
about what "gold cell" means (see ``hitab_lookup_multi`` in freeze_populations):

  answer   -- the ``[ANSWER]`` bucket of quantity_link. What the answer is made of.
  operand  -- ``gold_operands``, every quantity_link bucket. Phase 4's convention,
              and a superset: it also hands the reader the figure the QUESTION
              quotes.

Same reader, decoding and scoring as Phase 4 (Qwen2.5-7B-Instruct at the pinned
revision, 4-bit NF4, temp=0, seed=42, max_new_tokens=32, ``phase4_summary.em``).

  PYTHONPATH=. .venv/bin/python analysis/multicell_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import corpus_dump_vs_cell as cdv                                    # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from qwen_equiv_k import MODEL, REV, SYS                             # noqa: E402
from rag_agent.bench.hitab import (_table_offset, build_original_table,  # noqa: E402
                                   load_samples, load_table,
                                   resolve_gold_operands)
from rag_agent.llm.local_qwen import LocalQwenLLM                    # noqa: E402
from stratified_recall import build_chunks                           # noqa: E402

SEED, MAXNEW = 42, 32
SPLITS = ("dev", "test")
OUT = Path("results/multicell/ceiling_records.jsonl")


def answer_cells(data_dir, split, tids):
    """(query_id) -> {(tid, r, c)} from the ``[ANSWER]`` bucket alone.

    Resolved with the table-level offset that ``load_queries`` uses, so these
    coordinates land in the same data space as ``gold_operands``; passing the
    answer-only subset is safe because ``resolve_gold_operands`` re-validates
    the offset against the coords it is given.
    """
    samples = load_samples(data_dir, split)
    linked = defaultdict(list)
    for s in samples:
        linked[s.get("table_id")].append(s.get("linked_cells") or {})
    ot_cache, off_cache, out = {}, {}, {}
    for s in samples:
        tid = s.get("table_id")
        if tid not in tids:
            continue
        if tid not in ot_cache:
            raw = load_table(tid, data_dir)
            if raw is None:
                continue
            ot_cache[tid] = build_original_table(raw)
            off_cache[tid] = _table_offset(ot_cache[tid], linked[tid])
        lc = s.get("linked_cells") or {}
        bucket = (lc.get("quantity_link") or {}).get("[ANSWER]") or {}
        sub = {"quantity_link": {"[ANSWER]": bucket},
               "entity_link": lc.get("entity_link") or {}}
        ops = resolve_gold_operands(ot_cache[tid], sub, offset=off_cache[tid])
        out[s["id"]] = {(tid, o.row, o.col) for o in ops}
    return out


def main() -> int:
    dry = "--dry-run" in sys.argv
    import torch
    torch.manual_seed(SEED)

    bud = Budget("BAAI/bge-small-en-v1.5")
    jobs, missing = [], []
    for split in SPLITS:
        pop = f"hitab_{split}_lookup_multi"
        C = cdv.hitab_corpus("data/hitab", split, pop)
        by = defaultdict(dict)
        for n, (t, i, j) in enumerate(C.cell_owner):
            by[t][(i, j)] = n
        chunks, owner = build_chunks(C, "P4_path_cell", bud, by, "S3c")
        cell_text = {}
        for c, (tid, cells) in zip(chunks, owner):
            for (i, j) in cells:
                cell_text[(tid, i, j)] = c.text
        ans = answer_cells("data/hitab", split, set(C.tids))
        n_differ = 0
        for q in C.queries:
            conds = [("answer", ans.get(q["query_id"], set()))]
            # MEASURED 2026-09-01: on this frozen population the two sets are
            # identical for all 64 queries, so the operand leg would be 64
            # duplicate reader calls. It is still built when they diverge --
            # a re-freeze must not silently drop the comparison.
            if ans.get(q["query_id"], set()) != set(q["gold_cells"]):
                conds.append(("operand", set(q["gold_cells"])))
                n_differ += 1
            for cond, cells in conds:
                have = sorted(c for c in cells if c in cell_text)
                if len(have) != len(cells):
                    # a gold cell that is a header has no P4 sentence; record it
                    missing.append({"split": split, "query_id": q["query_id"],
                                    "cond": cond, "wanted": len(cells),
                                    "have": len(have)})
                jobs.append({"split": split, "pop": pop, "query_id": q["query_id"],
                             "cond": cond, "question": q["question"],
                             "gold_answer": str(q["answer"]),
                             "n_answer_cells": len(cells),
                             "cells": [list(c) for c in have],
                             "ctx": "\n".join(cell_text[c] for c in have)})

        print(f"[corpus] {split} n_q={len(C.queries)} chunks={len(chunks)} "
              f"answer_set!=operand_set: {n_differ}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["split"], r["cond"], r["query_id"]))
    print(f"[jobs] {len(jobs)} total, {len(done)} already done, "
          f"{len(missing)} with an unindexed gold cell", flush=True)

    if dry:
        json.dump(missing, open(OUT.parent / "ceiling_unindexed_cells.json", "w"), indent=1)
        return 0
    llm = LocalQwenLLM(model_name=MODEL, quantization="4bit")
    tok = llm.tokenizer
    fh = open(OUT, "a")
    t0 = time.time()
    for n, j in enumerate(jobs, 1):
        if (j["split"], j["cond"], j["query_id"]) in done:
            continue
        user = f"CONTEXT:\n{j['ctx']}\n\nQUESTION: {j['question']}\n\nAnswer:"
        ptok = len(tok(tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False)["input_ids"])
        t = time.time()
        raw = llm.complete(system=SYS, user=user, max_tokens=MAXNEW, temperature=0.0)
        lat = time.time() - t
        parsed = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        fh.write(json.dumps({
            k: j[k] for k in ("split", "pop", "query_id", "cond", "question",
                              "gold_answer", "n_answer_cells", "cells")} | {
            "n_cells_injected": len(j["cells"]),
            "pred_answer_raw": raw, "pred_parsed": parsed,
            "hit_token_cap": len(tok(raw, add_special_tokens=False)["input_ids"]) >= MAXNEW,
            "prompt_tokens": ptok, "latency_sec": round(lat, 3)}) + "\n")
        fh.flush()
        if n % 25 == 0:
            print(f"  {n}/{len(jobs)}  {(time.time()-t0)/60:.1f}min", flush=True)
    fh.close()
    json.dump(missing, open(OUT.parent / "ceiling_unindexed_cells.json", "w"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
