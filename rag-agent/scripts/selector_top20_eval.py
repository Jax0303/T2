#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retrieve top-20 (structural_leaf template, same hybrid alpha=0.7 as
sentence_disambiguation_eval.py), SELECT which of the 20 candidate sentences
answers the question, then answer using ONLY that one selected sentence as
context. No training. Selection methods (--method):
  llm           zero-shot: reader LLM outputs only the candidate number.
  llm_cot       same LLM, short chain-of-thought before an ANSWER: N line.
  cross_encoder BAAI/bge-reranker-v2-m3 scores all 20 (query, text) pairs,
                argmax picked; no LLM call for selection, reader still reads.
  hybrid_ce_llm cross_encoder narrows 20->5 by score, LLM zero-shot picks
                among those 5 (--k-shortlist).
  self_consistency  LLM zero-shot selection sampled --k-samples times at
                temperature 0.7, majority vote (ties -> lowest candidate #).
  order_ensemble  LLM zero-shot selection at temperature 0, run --k-samples
                times with the candidate ORDER independently reshuffled each
                time (not the sampling temperature) -- votes are tallied on
                the selected CELL, not the candidate number, since that
                number's meaning changes per shuffle. Targets the confirmed
                position bias (results/bottleneck_root_cause/, shuffled-order
                answer accuracy .6357 vs rank-order .7316, McNemar p<.000001):
                if the model's pick tracks CONTENT, it should win across
                independently-reshuffled orderings; if it tracks POSITION (as
                the bias measurement shows it partly does), it should scatter.
                self_consistency already tested temperature-only diversity at
                FIXED order and found no effect (.8000 == baseline) -- this is
                the order-diversity analog, untested until now.
  stuff         no selection: all k candidate texts concatenated into context,
                single reader call (the plain top-k "stuffing" baseline, for
                comparison against the selector methods above).

Same primary population as every other leg in this investigation: hitab test
primary (mode=all, m=1, aggregation=none), query count=991.

  PYTHONPATH=. .venv/bin/python scripts/selector_top20_eval.py --method llm_cot --limit 150
  PYTHONPATH=. .venv/bin/python scripts/selector_top20_eval.py --method cross_encoder --resume
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.eval.metrics import hitab_exact_match_text             # noqa: E402
from rag_agent.llm.factory import build_llm                           # noqa: E402
from rag_agent.retrieve.encoders import default_encoder               # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize        # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from bottleneck_diagnosis import load_primary_population              # noqa: E402
from sentence_disambiguation_eval import build_leaf_corpus, encode_corpus  # noqa: E402
from answer_accuracy import PROMPTS                                   # noqa: E402
import retrieval_accuracy as ra                                       # noqa: E402

OUT_DIR = ROOT / "results/selector_top20_20260916"
ALPHA = 0.7
K = 20

SELECTOR_SYSTEM = (
    "You are given a question and a numbered list of candidate table-cell "
    "facts. Exactly one candidate contains the value that answers the "
    "question. Reply with ONLY the number of that candidate and nothing else "
    "- no words, no punctuation."
)

SELECTOR_SYSTEM_COT = (
    "You are given a question and a numbered list of candidate table-cell "
    "facts. Exactly one candidate contains the value that answers the "
    "question. Briefly reason about which candidate matches (1-2 sentences), "
    "then end your reply with a final line of exactly the form "
    "'ANSWER: <number>'."
)


def parse_selection(text: str, k: int = K):
    m = re.search(r"\d+", text)
    if not m:
        return None
    n = int(m.group())
    return n if 1 <= n <= k else None


def parse_selection_cot(text: str, k: int = K):
    m = re.search(r"ANSWER:\s*(\d+)", text, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= k else None
    in_range = [int(x) for x in re.findall(r"\d+", text) if 1 <= int(x) <= k]
    return in_range[-1] if in_range else None  # fallback: last in-range number found


DEFAULT_OUT = {
    "llm": OUT_DIR / "selector_top20_qwen3_8b.jsonl",
    "llm_cot": OUT_DIR / "selector_top20_llm_cot.jsonl",
    "cross_encoder": OUT_DIR / "selector_top20_cross_encoder.jsonl",
    "hybrid_ce_llm": OUT_DIR / "selector_top20_hybrid_ce_llm.jsonl",
    "self_consistency": OUT_DIR / "selector_top20_self_consistency.jsonl",
    "order_ensemble": OUT_DIR / "selector_top20_order_ensemble.jsonl",
    "stuff": OUT_DIR / "selector_top20_stuff.jsonl",
}


def majority_vote(picks):
    """picks: list of int (1-based candidate #), some may be None. Ties -> lowest #."""
    valid = [p for p in picks if p is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    return min(n for n, c in counts.items() if c == best)


def majority_vote_by_cell(cell_votes, cand_cells):
    """cell_votes: list of cell tuples, one per reshuffled permutation (order_ensemble).
    Unlike majority_vote, ties can't break on the vote value itself -- a cell's
    candidate NUMBER differs across permutations, so the tie-break falls back to
    cand_cells' own (hybrid-rank) order, matching majority_vote's lowest-# rule."""
    counts = Counter(cell_votes)
    best = max(counts.values())
    tied = [c for c, n in counts.items() if n == best]
    return min(tied, key=lambda c: cand_cells.index(c))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit")
    ap.add_argument("--method", choices=list(DEFAULT_OUT), default="llm")
    ap.add_argument("--k", type=int, default=K, help="candidate pool size fed to the selector")
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N queries only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k-shortlist", type=int, default=5, help="hybrid_ce_llm: CE shortlist size")
    ap.add_argument("--k-samples", type=int, default=5, help="self_consistency: # sampled selections")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                     help="0 = present candidates in hybrid-rank order (default); "
                          "nonzero = shuffle per query (seeded, reproducible) to test position bias")
    ap.add_argument("--corpus", choices=["split", "gold"], default="split",
                     help="split (default, original behaviour) = rank candidates across all "
                          "538 split tables; gold = mask candidates to the query's own table "
                          "only (retrieval_accuracy.py's --corpus gold / TableRAG's per-table "
                          "setting) -- same masking formula as retrieval_accuracy.py main()")
    ap.add_argument("--template", choices=["structural_leaf"], default="structural_leaf",
                     help="cell-caption template for --arm cell")
    ap.add_argument("--arm", choices=["cell", "tablerag-leaf", "tablerag-path"], default="cell",
                     help="cell = structural_leaf atomic-cell candidates (needs a corpus-wide "
                          "embed pass); tablerag-leaf/-path = TableRAG's own per-table units "
                          "(scripts/retrieval_accuracy.py:tablerag_units), embedded on the fly per "
                          "query since gold scope only ever looks at one table. tablerag-* requires "
                          "--corpus gold (no split-corpus TableRAG path implemented -- not needed here).")
    ap.add_argument("--tablerag-dtype", default="infer", choices=["infer", "all_object"])
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--selftest", action="store_true",
                     help="no GPU/LLM: assert --corpus gold masking never lets a candidate "
                          "leak from a different table, for both --arm cell and tablerag-*")
    a = ap.parse_args()
    if a.arm != "cell" and a.corpus != "gold":
        raise SystemExit("--arm tablerag-* only implemented under --corpus gold")

    if a.selftest:
        pop = load_primary_population()
        qids = sorted(pop)[:5]
        enc = default_encoder()
        texts, coords = build_leaf_corpus(template_name="structural_leaf")
        coord_index = {c: i for i, c in enumerate(coords)}
        table_ids_arr = np.array([c[0] for c in coords])
        emb, _ = encode_corpus(texts, enc)
        bm = SparseBM25(_tokenize(t) for t in texts)
        for qid in qids:
            r = pop[qid]
            q = r["question"]
            qvec = enc.encode_query([q])[0].astype(np.float32)
            sel = np.flatnonzero(table_ids_arr == r["table_id"])
            dense, sparse = (emb @ qvec)[sel], bm.get_scores(_tokenize(q))[sel]
            hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
            order = sel[np.argsort(-hybrid, kind="stable")]
            topk = order[:K]
            assert all(coords[i][0] == r["table_id"] for i in topk), \
                f"{qid}: --corpus gold leaked a candidate from another table"
            gold = tuple(sorted(map(tuple, r["gold_cells"]))[0])
            assert coord_index[gold] in set(np.flatnonzero(table_ids_arr == r["table_id"])), \
                f"{qid}: gold cell not even in its own table's masked pool (bug in masking)"
            tab = hg.load_table(r["table_id"], a.data_dir)
            units = ra.tablerag_units(tab, tab.table, "leaf", "infer")
            for _, cells in units:
                assert all((r["table_id"], *rc) for rc in cells), "tablerag unit malformed"
        print(f"[selftest] OK -- {len(qids)} queries, --corpus gold never leaked cross-table, "
              "tablerag units well-formed")
        return 0
    if a.out:
        out_path = Path(a.out)
    else:
        name = DEFAULT_OUT[a.method].name
        if a.k != K:
            name = name.replace(f"top{K}", f"top{a.k}")
        if a.shuffle_seed:
            name = name.replace(".jsonl", f"_shuffle{a.shuffle_seed}.jsonl")
        out_path = OUT_DIR / name

    pop = load_primary_population()
    qids = sorted(pop)
    if a.limit:
        qids = qids[:a.limit]

    enc = default_encoder()
    table_ids_arr = None
    if a.arm == "cell":
        print(f"[corpus] building {a.template} corpus...", flush=True)
        texts, coords = build_leaf_corpus(template_name=a.template)
        coord_index = {c: i for i, c in enumerate(coords)}
        emb, cache_hit = encode_corpus(texts, enc)
        bm = SparseBM25(_tokenize(t) for t in texts)
        if a.corpus == "gold":
            table_ids_arr = np.array([c[0] for c in coords])
        print(f"[corpus] {len(texts)} units, embed_cache_hit={cache_hit}", flush=True)
    else:
        # tablerag-*: no global corpus -- each query only ever looks inside its
        # own table under --corpus gold, so build/embed that table's handful of
        # units on demand instead of embedding all 538 tables' TableRAG docs.
        texts = coords = emb = bm = coord_index = None
        print(f"[corpus] --arm {a.arm}: per-table units built on demand", flush=True)

    cross_encoder = None
    if a.method in ("cross_encoder", "hybrid_ce_llm"):
        from sentence_transformers import CrossEncoder                # noqa: E402
        print("[cross_encoder] loading BAAI/bge-reranker-v2-m3...", flush=True)
        cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    llm = build_llm(a.reader)
    import torch                                                      # noqa: E402
    torch.manual_seed(42)

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        raw = out.read_text(encoding="utf-8")
        whole = raw[:raw.rfind("\n") + 1] if raw else ""
        for line in whole.splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["query_id"]] = row
        if len(whole) != len(raw):
            out.write_text(whole, encoding="utf-8")
            print(f"[resume] truncated {len(raw) - len(whole)} trailing bytes", flush=True)
        if done and not a.resume:
            raise SystemExit(f"{out} has {len(done)} rows already — pass --resume")
        if done:
            print(f"[resume] continuing after {len(done)} rows", flush=True)

    t0 = time.time()
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        for idx, qid in enumerate(qids, 1):
            if qid in done:
                continue
            r = pop[qid]
            gold = tuple(sorted(map(tuple, r["gold_cells"]))[0])
            q = r["question"]
            qvec = enc.encode_query([q])[0].astype(np.float32)

            if a.arm == "cell":
                gidx = coord_index.get(gold)
                dense_full = emb @ qvec
                sparse_full = bm.get_scores(_tokenize(q))
                if a.corpus == "gold":
                    sel = np.flatnonzero(table_ids_arr == r["table_id"])
                    dense, sparse = dense_full[sel], sparse_full[sel]
                    hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
                    order = sel[np.argsort(-hybrid, kind="stable")]
                else:
                    hybrid = ALPHA * _minmax(dense_full) + (1 - ALPHA) * _minmax(sparse_full)
                    order = np.argsort(-hybrid, kind="stable")
                # rank against the FULL corpus index space either way (sentinel
                # for units outside this query's --corpus gold table, same as
                # "not in range" -> gold_rank stays None-equivalent for those).
                ranks = np.full(len(texts), len(texts) + 1, dtype=np.int64)
                ranks[order] = np.arange(1, len(order) + 1)
                gold_rank = int(ranks[gidx]) if gidx is not None else None
                gold_in_topk = bool(gold_rank is not None and gold_rank <= a.k)

                topk_idx = order[:a.k]
                cand_cells = [coords[i] for i in topk_idx]
                cand_texts = [texts[i] for i in topk_idx]
            else:
                mode = "leaf" if a.arm == "tablerag-leaf" else "path"
                tab = hg.load_table(r["table_id"], a.data_dir)
                t = tab.table
                units = ra.tablerag_units(tab, t, mode, a.tablerag_dtype)
                texts_t = [u[0] for u in units]
                covers_t = [frozenset((r["table_id"], i, j) for i, j in u[1]) for u in units]
                # per-table corpora are tiny (tens of docs) -- no cache needed, reuse `enc`
                dense_t = enc.encode(texts_t) @ qvec
                bm_t = SparseBM25(_tokenize(x) for x in texts_t)
                sparse_t = bm_t.get_scores(_tokenize(q))
                hybrid_t = ALPHA * _minmax(dense_t) + (1 - ALPHA) * _minmax(sparse_t)
                order_t = np.argsort(-hybrid_t, kind="stable")
                selection = ra.budget_select(order_t, covers_t, texts_t, budget=20, dump=1)
                cand_texts = [u["text"] for u in selection.units]
                cand_cells = [gold if gold in set(u["cells"]) else
                             (tuple(u["cells"][0]) if u["cells"] else ("__none__", -1, -1))
                             for u in selection.units]
                gold_rank = next((i + 1 for i, c in enumerate(cand_cells) if c == gold), None)
                gold_in_topk = gold_rank is not None

            if a.shuffle_seed:
                perm = list(range(len(cand_cells)))
                random.Random(f"{a.shuffle_seed}:{qid}").shuffle(perm)
                cand_cells = [cand_cells[i] for i in perm]
                cand_texts = [cand_texts[i] for i in perm]
            gold_list_pos = cand_cells.index(gold) + 1 if gold in cand_cells else None

            ce_gold_in_shortlist = None
            if a.method == "stuff":
                sel_raw = sel_n = None
                selected_cell = None
                selected_text = "\n".join(cand_texts)
                selected_is_gold = None
            elif a.method == "cross_encoder":
                scores = cross_encoder.predict([(q, t) for t in cand_texts])
                sel_pos = int(np.argmax(scores))
                sel_raw, sel_n = None, sel_pos + 1
            elif a.method == "hybrid_ce_llm":
                scores = cross_encoder.predict([(q, t) for t in cand_texts])
                short_idx = list(np.argsort(-scores)[:a.k_shortlist])
                short_cells = [cand_cells[i] for i in short_idx]
                short_texts = [cand_texts[i] for i in short_idx]
                ce_gold_in_shortlist = bool(gold in [tuple(c) for c in short_cells])
                numbered = "\n".join(f"{n}. {t}" for n, t in enumerate(short_texts, 1))
                sel_user = (f"Question: {q}\n\nCandidates:\n{numbered}\n\n"
                           f"Which candidate number answers the question?")
                sel_raw = llm.complete(SELECTOR_SYSTEM, sel_user, max_tokens=8, temperature=0.0)
                sel_n = parse_selection(sel_raw, len(short_texts))
                local_pos = (sel_n - 1) if sel_n else 0
                sel_pos = short_idx[local_pos]
            elif a.method == "self_consistency":
                numbered = "\n".join(f"{n}. {t}" for n, t in enumerate(cand_texts, 1))
                sel_user = (f"Question: {q}\n\nCandidates:\n{numbered}\n\n"
                           f"Which candidate number answers the question?")
                picks = []
                for _ in range(a.k_samples):
                    raw_i = llm.complete(SELECTOR_SYSTEM, sel_user, max_tokens=8, temperature=0.7)
                    picks.append(parse_selection(raw_i, len(cand_texts)))
                sel_n = majority_vote(picks)
                sel_raw = json.dumps(picks)
                sel_pos = (sel_n - 1) if sel_n else 0
            elif a.method == "order_ensemble":
                cell_votes = []  # cell tuples, one per reshuffled permutation
                for p in range(a.k_samples):
                    perm = list(range(len(cand_texts)))
                    random.Random(f"order_ensemble:{qid}:{p}").shuffle(perm)
                    perm_texts = [cand_texts[i] for i in perm]
                    perm_cells = [cand_cells[i] for i in perm]
                    numbered = "\n".join(f"{n}. {t}" for n, t in enumerate(perm_texts, 1))
                    sel_user = (f"Question: {q}\n\nCandidates:\n{numbered}\n\n"
                               f"Which candidate number answers the question?")
                    raw_i = llm.complete(SELECTOR_SYSTEM, sel_user, max_tokens=8, temperature=0.0)
                    n_i = parse_selection(raw_i, len(perm_texts))
                    cell_votes.append(tuple(perm_cells[n_i - 1] if n_i else perm_cells[0]))
                winner = majority_vote_by_cell(cell_votes, cand_cells)
                sel_pos = cand_cells.index(winner)
                sel_n = sel_pos + 1
                sel_raw = json.dumps([list(v) for v in cell_votes])
            else:
                numbered = "\n".join(f"{n}. {t}" for n, t in enumerate(cand_texts, 1))
                sel_user = (f"Question: {q}\n\nCandidates:\n{numbered}\n\n"
                           f"Which candidate number answers the question?")
                if a.method == "llm_cot":
                    sel_raw = llm.complete(SELECTOR_SYSTEM_COT, sel_user, max_tokens=200, temperature=0.0)
                    sel_n = parse_selection_cot(sel_raw, len(cand_texts))
                else:
                    sel_raw = llm.complete(SELECTOR_SYSTEM, sel_user, max_tokens=8, temperature=0.0)
                    sel_n = parse_selection(sel_raw, len(cand_texts))
                sel_pos = (sel_n - 1) if sel_n else 0  # unparseable -> fall back to rank-1 candidate
            if a.method != "stuff":
                selected_cell = cand_cells[sel_pos]
                selected_text = cand_texts[sel_pos]
                selected_is_gold = bool(tuple(selected_cell) == gold)

            reader_user = f"Context:\n{selected_text}\n\nQuestion: {q}\nAnswer:"
            pred = llm.complete(PROMPTS["neutral"], reader_user, max_tokens=64, temperature=0.0)
            ok = hitab_exact_match_text(pred, r["answer"])

            row = {
                "query_id": qid, "gold_cell": list(gold), "gold_rank": gold_rank,
                "gold_in_topk": int(gold_in_topk), "k": a.k, "k_actual": len(cand_texts),
                "gold_list_pos": gold_list_pos, "arm": a.arm, "corpus": a.corpus,
                "shuffle_seed": a.shuffle_seed,
                "selector_raw": sel_raw, "selector_parsed": sel_n,
                "selector_fallback_used": (a.method != "stuff" and sel_n is None),
                "selected_cell": list(selected_cell) if selected_cell else None,
                "selected_is_gold": selected_is_gold if selected_is_gold is None else int(selected_is_gold),
                "ce_gold_in_shortlist": ce_gold_in_shortlist,
                "pred": pred, "answer": r["answer"], "answer_correct": int(ok),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if idx % 25 == 0 or idx == len(qids):
                done_rows = idx - len(done) if idx <= len(qids) else idx
                print(f"  {idx}/{len(qids)}  {time.time() - t0:.0f}s", flush=True)

    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
