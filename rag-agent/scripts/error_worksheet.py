"""Dump every wrong answer of a finished run to one spreadsheet.

Answers one question the aggregate numbers cannot: when the reader is wrong, WAS
THE ANSWER EVEN IN THE CONTEXT? The run records keep only ``osc`` and ``pred``,
so a wrong answer at osc=1 and a wrong answer at osc=0 look the same in a
summary and are opposite failures -- one is the reader, one is retrieval.

Rebuilds the run's corpus, index and per-query context from the SAME code path
``corpus_dump_vs_cell`` uses, then asserts the replayed ``tokens``/``osc`` match
what is on disk before writing anything. A replay that drifted from the driver
would produce a worksheet describing a run that never happened, so the mismatch
count is printed and a nonzero one is fatal unless ``--allow-drift``.

Sheets:
  errors     one row per wrong query -- question, gold, prediction, retrieval
  context    one row per sentence ACTUALLY SENT, with its rank and its scores
  gold_cells one row per gold cell, whether it was retrieved and at what rank
  emb_query / emb_gold   the raw vectors, one column per dimension
  repeats    only with --reader: the same prompt re-read N times
  meta       the configuration this worksheet was replayed under

The vectors also go to a ``.npz`` beside the workbook, because 384 columns of
float in a cell grid is for reading, not for arithmetic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus_dump_vs_cell import (ALPHA, FROZEN_POP, RHB_POPS, Corpus,
                                 _CachedEncoder, aitqa_corpus, hitab_corpus,
                                 osc_share, positions, rank_of,
                                 realhitbench_corpus, table_index_text)
from baseline_comparison_llm import Budget
from point3_reconstruction_cost import cell_text
from rag_agent.retrieve.encoders import default_encoder
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax
from rag_agent.serialization.base import Chunk
from rag_agent.serialization.caption import caption_sentence
from rag_agent.eval.metrics import _to_nums, numeric_match
from rag_agent.serialization.templates import (MT2NET, STRUCTURAL,
                                               STRUCTURAL_COMPACT)

XL_MAX = 32000          # Excel's hard cell limit is 32,767 characters


def error_class(pred, gold) -> str:
    """Which KIND of wrong the answer is, so the sheet sorts by cause.

    ``sign_only`` and ``scale_only`` are notation, not reading: every digit is
    right. They matter because HiTab's official scorer counts them as wrong
    while the RealHiTBench and MixRAG scorers forgive them, so a number compared
    across those papers has to state how many of these it is carrying.
    """
    p, g = _to_nums(str(pred)), _to_nums(str(gold))
    if not g:
        return "gold_not_numeric"
    if not p:
        return "pred_not_numeric"
    pv, gv = p[0], g[0]
    if abs(abs(pv) - abs(gv)) < 1e-6:
        return "sign_only"
    for m in (100, 0.01, 1000, 0.001, 1e6, 1e-6):
        if abs(abs(pv) - abs(gv * m)) < 1e-6 * max(1.0, abs(gv * m)):
            return "scale_only"
    if numeric_match(str(pred), [gv], rel_tol=0.02):
        return "within_2pct"
    return "different_value"


def pred_source(pred, sentences) -> str:
    """Did the reader COPY a number out of its context, or invent one?

    A wrong value that is present in the context is a selection failure -- the
    right kind of cell was there and the wrong one was picked. A wrong value
    that appears nowhere in the context is arithmetic or hallucination, and no
    amount of retrieval fixes it.
    """
    p = _to_nums(str(pred))
    if not p:
        return "no_number"
    vals = set()
    for t in sentences:
        vals.update(_to_nums(str(t)))
    return ("value_present_in_context"
            if any(abs(p[0] - v) < 1e-6 for v in vals)
            else "value_not_in_context")


def render_cells(C: Corpus, scheme: str, no_title: bool) -> None:
    """Apply ``--cell-scheme`` to ``C.cell_text`` exactly as the driver does."""
    if scheme == "S2":
        return                                   # already what the builder made
    if scheme == "S2r":
        C.cell_text[:] = [cell_text(rp[::-1], cp[::-1], v, "S2")
                          for rp, cp, v in C.cell_paths]
        return
    if scheme in ("S3", "S3c", "mt2net"):
        tmpl = {"mt2net": MT2NET, "S3c": STRUCTURAL_COMPACT}.get(scheme, STRUCTURAL)
        C.cell_text[:] = [
            caption_sentence("" if no_title else C.title.get(t, ""),
                             rp, cp, v, template=tmpl)
            for (rp, cp, v), (t, _i, _j) in zip(C.cell_paths, C.cell_owner)]
        return
    raise SystemExit(f"--cell-scheme {scheme} is not replayable here; the "
                     f"tag/structural variants rewrite the corpus")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="the run's *_records.jsonl")
    ap.add_argument("--arm", default="cell")
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "aitqa", "realhitbench"])
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--cell-scheme", default="S3c")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--retriever", default="dense", choices=list(ALPHA))
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--out", required=True, help="path to the .xlsx to write")
    ap.add_argument("--allow-drift", action="store_true",
                    help="write the workbook even if the replay disagrees with "
                         "the records -- for inspecting the disagreement, never "
                         "for a number that goes in the paper")
    ap.add_argument("--also-correct", action="store_true",
                    help="include right answers too, flagged, as the control")
    ap.add_argument("--reader", default="",
                    help="re-read each wrong context with this LLM (needs a GPU "
                         "for the local spec); empty = leave the repeats sheet out")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="the recorded runs are greedy (do_sample=False), so "
                         "repeating them is a no-op by construction. This samples "
                         "instead, which is what makes the spread meaningful; "
                         "pass 0 to prove the determinism rather than measure it")
    ap.add_argument("--answer-mode", default="direct", choices=["direct", "codegen"])
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.records).read_text().splitlines() if l.strip()]
    by_qid = {r["query_id"]: r for r in recs}
    if args.arm not in recs[0]:
        raise SystemExit(f"--arm {args.arm} is not in {args.records} "
                         f"(has {[k for k, v in recs[0].items() if isinstance(v, dict)]})")

    if args.dataset == "hitab":
        C = hitab_corpus(args.data_dir, args.split, args.population)
    elif args.dataset == "aitqa":
        C = aitqa_corpus()
    else:
        pop_name = args.population if args.population in RHB_POPS \
            else FROZEN_POP["realhitbench"]
        C = realhitbench_corpus(pin=pop_name, **RHB_POPS[pop_name])
    render_cells(C, args.cell_scheme, args.no_title)

    alpha = ALPHA[args.retriever]
    enc = _CachedEncoder(default_encoder(model_name=args.embed_model),
                         args.cache_dir,
                         f"{args.dataset}_{args.split}_{args.embed_model}")
    cell_chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=txt,
                         scheme=args.cell_scheme, kind="cell")
                   for txt, (t, i, j) in zip(C.cell_text, C.cell_owner)]
    ix = HybridIndex(cell_chunks, encoder=enc, alpha=0.5)
    bud = Budget(args.embed_model)
    tok = np.array([bud.count(c.text) for c in cell_chunks], dtype=np.int32)
    pos_of_cell = {k: n for n, k in enumerate(C.cell_owner)}
    emb = ix._emb                                    # the corpus vectors, as used

    llm = None
    if args.reader:
        from rag_agent.llm.factory import build_llm
        llm = build_llm(args.reader)

    from rag_agent.generate.answerer import _DIRECT_SYS, _CODEGEN_SYS
    from rag_agent.eval.metrics import hitab_exact_match_text

    err_rows, ctx_rows, gold_rows, rep_rows = [], [], [], []
    qvecs, gvecs, gvec_keys = [], [], []
    drift = 0
    for q in C.queries:
        r = by_qid.get(q["query_id"])
        if r is None:
            continue
        a = r[args.arm]
        wrong = not a.get("answer_em")
        if not wrong and not args.also_correct:
            continue

        order = rank_of(ix, q["question"], alpha)
        rank = positions(order)
        bm = ix._bm25_scores(q["question"])
        dn = ix._dense_scores(q["question"])
        fused = alpha * _minmax(dn) + (1 - alpha) * _minmax(bm)
        qv = enc.inner.encode([q["question"]])[0]

        used, parts, in_ctx, sent = 0, [], set(), []
        for pos in order:
            n = int(tok[pos])
            if used + n > args.budget:
                if used + int(tok.min()) > args.budget:
                    break
                continue
            used += n
            parts.append(C.cell_text[pos])
            in_ctx.add(C.cell_owner[pos])
            sent.append(pos)
        gold_cells = q["gold_cells"]
        osc_v, per_cell_v = osc_share(gold_cells & in_ctx, gold_cells)
        if used != a["tokens"] or osc_v != a["osc"]:
            drift += 1

        top1 = order[0]
        gpos = [pos_of_cell[g] for g in sorted(gold_cells) if g in pos_of_cell]
        cos_gold = [float(emb[p] @ qv) for p in gpos]
        err_rows.append({
            "query_id": q["query_id"], "wrong": int(wrong),
            "question": q["question"],
            "gold_answer": " | ".join(str(x) for x in q["answer"]),
            "prediction": str(a.get("pred", ""))[:XL_MAX],
            "answer_em": a.get("answer_em"),
            "error_class": error_class(a.get("pred", ""), q["answer"]),
            "pred_source": pred_source(a.get("pred", ""), parts),
            "osc": a["osc"], "per_cell": a["per_cell"],
            "gold_table_any": a["gold_table_any"],
            "table_rank": r.get("table_rank"),
            "tokens": a["tokens"], "n_tables": a["n_tables"],
            "n_gold_cells": len(gold_cells),
            "n_gold_in_context": len(gold_cells & in_ctx),
            "best_gold_rank": (min(int(rank[p]) for p in gpos) + 1) if gpos else None,
            "worst_gold_rank": (max(int(rank[p]) for p in gpos) + 1) if gpos else None,
            "n_context_sentences": len(sent),
            "top1_text": C.cell_text[top1],
            "cos_q_top1": float(emb[top1] @ qv),
            "cos_q_gold_max": max(cos_gold) if cos_gold else None,
            "cos_q_gold_min": min(cos_gold) if cos_gold else None,
            "gold_table": q["gold_table"],
            "gold_table_md": "\n".join(C.md_lines[q["gold_table"]])[:XL_MAX],
        })
        qvecs.append(qv)
        for k, pos in enumerate(sent):
            t, i, j = C.cell_owner[pos]
            ctx_rows.append({
                "query_id": q["query_id"], "context_order": k + 1,
                "corpus_rank": int(rank[pos]) + 1, "table_id": t, "row": i, "col": j,
                "is_gold": int((t, i, j) in gold_cells),
                "sentence": C.cell_text[pos], "tokens": int(tok[pos]),
                "bm25": float(bm[pos]), "dense": float(dn[pos]),
                "fused": float(fused[pos]), "cos_q_sentence": float(emb[pos] @ qv),
            })
        for g in sorted(gold_cells):
            p = pos_of_cell.get(g)
            gold_rows.append({
                "query_id": q["query_id"], "table_id": g[0], "row": g[1], "col": g[2],
                "sentence": C.cell_text[p] if p is not None else "",
                "in_context": int(g in in_ctx),
                "corpus_rank": (int(rank[p]) + 1) if p is not None else None,
                "bm25": float(bm[p]) if p is not None else None,
                "dense": float(dn[p]) if p is not None else None,
                "cos_q_sentence": float(emb[p] @ qv) if p is not None else None,
            })
            if p is not None:
                gvecs.append(emb[p])
                gvec_keys.append((q["query_id"], *map(str, g)))

        if llm is not None and wrong:
            sysmsg = _CODEGEN_SYS if args.answer_mode == "codegen" else _DIRECT_SYS
            head = "ROWS" if args.answer_mode == "codegen" else "CONTEXT"
            tail = ("One line: answer = ..." if args.answer_mode == "codegen"
                    else "Answer:")
            user = f"{head}:\n" + "\n".join(parts) + \
                   f"\n\nQUESTION: {q['question']}\n\n{tail}"
            for t_ in range(args.repeat):
                out = llm.complete(system=sysmsg, user=user, max_tokens=512,
                                   temperature=args.temperature)
                rep_rows.append({
                    "query_id": q["query_id"], "trial": t_ + 1,
                    "temperature": args.temperature,
                    "prediction": str(out)[:XL_MAX],
                    "answer_em": int(bool(hitab_exact_match_text(out, q["answer"]))),
                    "recorded_prediction": str(a.get("pred", ""))[:XL_MAX],
                })
        if len(err_rows) % 50 == 0:
            print(f"  {len(err_rows)} rows", flush=True)

    print(f"[replay] {drift} of {len(err_rows)} queries disagree with the records")
    if drift and not args.allow_drift:
        raise SystemExit("replay drifted from the recorded run -- refusing to "
                         "write. Check --cell-scheme/--budget/--retriever, or "
                         "pass --allow-drift to inspect it")

    import pandas as pd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dim = emb.shape[1]
    sheets = {
        "errors": pd.DataFrame(err_rows),
        "context": pd.DataFrame(ctx_rows),
        "gold_cells": pd.DataFrame(gold_rows),
        "emb_query": pd.concat(
            [pd.DataFrame({"query_id": [r["query_id"] for r in err_rows]}),
             pd.DataFrame(np.array(qvecs), columns=[f"d{k}" for k in range(dim)])],
            axis=1),
        "emb_gold": pd.concat(
            [pd.DataFrame(gvec_keys, columns=["query_id", "table_id", "row", "col"]),
             pd.DataFrame(np.array(gvecs), columns=[f"d{k}" for k in range(dim)])],
            axis=1) if gvecs else pd.DataFrame(),
        "meta": pd.DataFrame([{
            "records": args.records, "arm": args.arm, "dataset": args.dataset,
            "population": args.population, "cell_scheme": args.cell_scheme,
            "budget": args.budget, "retriever": args.retriever, "alpha": alpha,
            "embed_model": args.embed_model, "encoder_name": enc.inner.name,
            "embed_dim": dim, "n_corpus_cells": len(cell_chunks),
            "n_queries_in_records": len(recs), "n_rows": len(err_rows),
            "replay_drift": drift, "reader": args.reader or "(not re-read)",
            "repeat": args.repeat if llm else 0, "temperature": args.temperature,
        }]),
    }
    if rep_rows:
        sheets["repeats"] = pd.DataFrame(rep_rows)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if len(df):
                df.to_excel(xw, sheet_name=name, index=False)
    np.savez_compressed(out.with_suffix(".npz"),
                        query=np.array(qvecs),
                        query_ids=np.array([r["query_id"] for r in err_rows]),
                        gold=np.array(gvecs) if gvecs else np.zeros((0, dim)),
                        gold_keys=np.array(gvec_keys) if gvec_keys else np.zeros((0, 4)))
    print(f"[write] {out}  errors={len(err_rows)} context={len(ctx_rows)} "
          f"gold={len(gold_rows)} repeats={len(rep_rows)}")
    print(f"[write] {out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
