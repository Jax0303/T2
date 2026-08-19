# SPDX-License-Identifier: MIT
"""MT2Net's retriever, trained and run inside this repo's harness.

Zhao et al. (2022) §4 retrieve supporting facts by concatenating the question
with each candidate sentence and scoring it with a BERT-base **binary
classifier**, keeping the top-n. Every retrieval number in this repo so far came
from BM25 / dense / an off-the-shelf cross-encoder instead, so "their retriever
vs ours" has never been measured — only "their retriever's published recall on
their own population", which is not comparable (different unit, different
population, different gold).

This script closes that: identical eval population, identical within-document
pool, identical gold operand cells, identical cell sentences (the ``mt2net``
template), and the *only* thing that changes is the scorer.

Two honest limitations, both to be stated wherever a number from here is cited:

1. **Less training data than the original.** MT2Net trains on the full train
   split (7,830 QA, all question types, text facts included). Here the training
   pool is the table-only arithmetic multi-operand slice that is disjoint from
   the eval queries — ~1.4k questions. A classifier trained on ~17% of the data
   is a weaker classifier, so a win over it is partly a data-budget artefact.
2. **The mt2net template readings are provisional** (one published example) —
   see ``rag_agent.serialization.templates``.

Train/eval separation: the eval population is the first ``--eval-queries``
qualifying queries (the same slice every other within-doc result on disk uses);
training draws only from the queries after that cut, and their documents are
built into a separate corpus, so no eval cell is ever a training candidate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rag_agent.runenv import run_env

from operand_collision_multihiertt import build_corpus, cell_text, load_population
from standard_ir_metrics_from_records import (hit_at_k, ndcg_at_k, rr,
                                              recall_at_k, set_em_at_k)

KS = (1, 5, 10, 20, 50)


def summarize(per_query: dict) -> dict:
    """Byte-identical metric set to operand_collision_within_doc.summarize."""
    qs = sorted(per_query)
    n = len(qs)
    out = {"n_queries": n, "mrr": round(sum(rr(per_query[q]) for q in qs) / n, 4)}
    for k in KS:
        out[f"hit@{k}"] = round(sum(hit_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"recall@{k}"] = round(sum(recall_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"ndcg@{k}"] = round(sum(ndcg_at_k(per_query[q], k) for q in qs) / n, 4)
        out[f"set_em@{k}"] = round(sum(set_em_at_k(per_query[q], k) for q in qs) / n, 4)
    return out


def fill_by_index(order, texts, count, limit):
    """Greedily take units in rank order while they fit, returning WHICH ones.

    Same rule as ``Budget.fill`` -- a unit too long to fit is skipped, not a
    stopping point, so one fat cell cannot shut out every shorter one behind
    it. Budget.fill returns the texts; OSC needs the indices, which is the only
    reason this exists.
    """
    kept, used = [], 0
    for gi in order:
        n = count(texts[gi])
        if used + n > limit:
            continue
        kept.append(gi)
        used += n
    return kept, used


def pools(pop, cells):
    """query uid -> global indices of every cell in that query's own document."""
    by_uid = {}
    for gi, c in enumerate(cells):
        by_uid.setdefault(c["table"][0], []).append(gi)
    return {q["uid"]: by_uid[q["uid"]] for q in pop}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-queries", type=int, default=300)
    ap.add_argument("--train-queries", type=int, default=1361)
    ap.add_argument("--scheme", default="mt2net",
                    help="cell sentence template both arms are scored over")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--neg-ratio", type=int, default=20,
                    help="sampled non-gold cells per gold cell, same document")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", default="",
                    help="directory to write the fine-tuned classifier to; the "
                         "cross-dataset transfer leg needs these weights, and "
                         "retraining to get them back costs a GPU-quarter-hour")
    ap.add_argument("--out", default="results/mt2net_retriever_baseline.json")
    ap.add_argument("--population", default="arith_multi",
                    choices=["arith_multi", "lookup_single"],
                    help="lookup_single exists because the standardised reader "
                         "is a 4-bit local 7B that scores .02-.05 on arithmetic "
                         "(corpus_dump_vs_cell_reader_dense_*): an answer leg on "
                         "arith_multi measures the reader, not the retriever")
    ap.add_argument("--reader", default="",
                    help="answer the questions too (e.g. "
                         "local:Qwen/Qwen2.5-7B-Instruct). Both arms get the same "
                         "pool, budget, prompt and scorer, so an EM difference is "
                         "a difference in what the RETRIEVER put in the window")
    ap.add_argument("--budget", type=int, default=512,
                    help="reader context tokens per arm")
    ap.add_argument("--ours-scheme", default="S3",
                    help="our arm's deployed index unit, against mt2net's own "
                         "template. System-level: each retriever gets the cell "
                         "text it actually ships")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = run_env(args.seed, args.model)

    total = args.eval_queries + args.train_queries
    queries, docs = load_population(total, args.population)
    ev_q, tr_q = queries[:args.eval_queries], queries[args.eval_queries:]
    ev_uids = {q["uid"] for q in ev_q}
    tr_q = [q for q in tr_q if q["uid"] not in ev_uids]      # no shared documents

    _, ev_cells, ev_pop = build_corpus(ev_q, {q["uid"]: docs[q["uid"]] for q in ev_q})
    print(f"[eval ] {len(ev_pop)} queries | {len(ev_cells)} cells", flush=True)

    _, tr_cells, tr_pop = build_corpus(tr_q, {q["uid"]: docs[q["uid"]] for q in tr_q})
    print(f"[train] {len(tr_pop)} queries | {len(tr_cells)} cells", flush=True)

    # --- training pairs: gold cells positive, sampled same-document cells negative
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tr_texts = [cell_text(c, args.scheme) for c in tr_cells]
    tr_pools = pools(tr_pop, tr_cells)
    examples = []                                   # (question, sentence, label)
    for q in tr_pop:
        gold = {int(g) for g in q["gold"]}
        pool = [gi for gi in tr_pools[q["uid"]] if gi not in gold]
        negs = rng.sample(pool, min(len(pool), args.neg_ratio * len(gold)))
        examples += [(q["question"], tr_texts[gi], 1.0) for gi in gold]
        examples += [(q["question"], tr_texts[gi], 0.0) for gi in negs]
    rng.shuffle(examples)
    pos = sum(1 for e in examples if e[2] == 1.0)
    print(f"[pairs] {len(examples)} ({pos} pos / {len(examples)-pos} neg)", flush=True)

    # Plain torch loop rather than Trainer: keeps `accelerate` out of a research
    # env whose every result file pins its dependency versions.
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1).to(dev)

    def encode(batch):
        return tok([q for q, _, _ in batch], [s for _, s, _ in batch],
                   padding=True, truncation=True, max_length=args.max_length,
                   return_tensors="pt").to(dev)

    steps = args.epochs * ((len(examples) + args.batch_size - 1) // args.batch_size)
    warm = max(1, int(0.1 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
    lossf = torch.nn.BCEWithLogitsLoss()

    t0, model_step = time.time(), 0
    model.train()
    for ep in range(args.epochs):
        rng.shuffle(examples)
        run_loss = 0.0
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            y = torch.tensor([b[2] for b in batch], device=dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                logits = model(**encode(batch)).logits.squeeze(-1)
                loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            model_step += 1
        print(f"[fit  ] epoch {ep+1}/{args.epochs} "
              f"loss={run_loss / max(1, model_step):.4f} ({time.time()-t0:.0f}s)", flush=True)
    train_s = round(time.time() - t0, 1)

    if args.save_model:
        Path(args.save_model).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_model)
        tok.save_pretrained(args.save_model)
        print(f"[save ] {args.save_model}", flush=True)

    @torch.no_grad()
    def predict(pairs):
        model.eval()
        out = []
        for i in range(0, len(pairs), args.batch_size):
            b = [(q, s, 0.0) for q, s in pairs[i:i + args.batch_size]]
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                out.append(model(**encode(b)).logits.squeeze(-1).float().cpu())
        return torch.cat(out).numpy()

    # --- eval: score every cell of the query's own document, rank by P(relevant)
    ev_texts = [cell_text(c, args.scheme) for c in ev_cells]
    ev_pools = pools(ev_pop, ev_cells)

    # --- answer leg: their retriever and ours, same pool, budget, prompt, scorer.
    # Each arm reads the cell text it actually ships, so this is a system-level
    # comparison of "what did the retriever put in the window", not a template
    # ablation -- that one is operand_collision_within_doc's job.
    llm = bud = None
    if args.reader:
        from rag_agent.eval.metrics import hitab_exact_match_text
        from rag_agent.generate.answerer import _DIRECT_SYS
        from rag_agent.llm.factory import build_llm
        from rag_agent.retrieve.encoders import default_encoder
        from baseline_comparison_llm import Budget

        llm, bud = build_llm(args.reader), Budget()
        ours_texts = [cell_text(c, args.ours_scheme) for c in ev_cells]
        ours_enc = default_encoder()
        ours_emb = ours_enc.encode(ours_texts)
        ours_emb = ours_emb / (np.linalg.norm(ours_emb, axis=1, keepdims=True) + 1e-9)
        q_emb = ours_enc.encode([q["question"] for q in ev_pop])
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

        def fill(order, texts):
            return fill_by_index(order, texts, bud.count, args.budget)

        def answer(kept, texts, question, gold_answer):
            """Same prompt frame as corpus_dump_vs_cell, so EM is on one ruler."""
            user = ("CONTEXT:\n" + "\n".join(texts[gi] for gi in kept)
                    + f"\n\nQUESTION: {question}\n\nAnswer:")
            out_txt = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=512)
            if not out_txt and llm.last_finish_reason == "length":
                out_txt = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=1024)
            return int(bool(hitab_exact_match_text(out_txt, gold_answer))), out_txt[:120]

    per_query, records, ans = {}, [], []
    for qi, q in enumerate(ev_pop):
        idxs = ev_pools[q["uid"]]
        gold = {int(g) for g in q["gold"]}
        scores = predict([(q["question"], ev_texts[gi]) for gi in idxs])
        mt_order = [idxs[int(l)] for l in np.argsort(-np.asarray(scores))]
        rank_of = {}
        for pos_, gi in enumerate(mt_order, 1):
            if gi in gold:
                rank_of[gi] = pos_
                if len(rank_of) == len(gold):
                    break
        per_query[qi] = [rank_of.get(g) for g in gold]
        for g in gold:
            records.append({"scheme": args.scheme, "retriever": "mt2net_bert",
                            "query": qi, "cell": g, "rank": rank_of.get(g)})
        if llm is not None:
            sims = ours_emb[idxs] @ q_emb[qi]
            ours_order = [idxs[int(l)] for l in np.argsort(-sims)]
            gold_answer = docs[q["uid"]]["answer"]
            rec = {"query": qi, "uid": q["uid"], "gold": sorted(gold),
                   "answer": gold_answer}
            for arm, order, txts in (("mt2net_bert", mt_order, ev_texts),
                                     ("ours", ours_order, ours_texts)):
                kept, used = fill(order, txts)
                em, pred = answer(kept, txts, q["question"], gold_answer)
                rec[arm] = {"answer_em": em, "tokens": used, "pred": pred,
                            "n_cells": len(kept), "osc": int(gold <= set(kept))}
            ans.append(rec)
            if len(ans) % 25 == 0:
                print(f"  [answer] {len(ans)}/{len(ev_pop)}", flush=True)

    out = {
        "env": env,
        "leg": "MT2Net's BERT-base binary-classifier retriever, trained here and "
               "evaluated on the same population/pool/gold/sentences as "
               "operand_collision_within_doc.py, so the scorer is the only variable",
        "population": {"name": f"multihiertt_{args.population}_within_doc",
                       "n_queries": len(ev_pop),
                       "pool": "within-document (all tables of the query's own doc)"},
        "train": {"n_queries": len(tr_pop), "n_pairs": len(examples),
                  "n_positive": pos, "neg_ratio": args.neg_ratio,
                  "epochs": args.epochs, "lr": args.lr, "train_s": train_s,
                  "disjoint_from_eval": "by document uid"},
        "scheme": args.scheme,
        "model": args.model,
        "caveat": "trained on the table-only arithmetic slice (~1.4k questions), not "
                  "the full 7,830-QA train split MT2Net uses; a weaker classifier "
                  "than the original by data budget alone",
        "by_scheme": {args.scheme: {"mt2net_bert": summarize(per_query)}},
    }
    if ans:
        from manual_sentence_ceiling import mcnemar
        agg = lambda a, k: round(float(np.mean([r[a][k] for r in ans])), 4)
        out["answer_leg"] = {
            "reader": llm.name, "budget_tokens": args.budget,
            "ours_scheme": args.ours_scheme, "n": len(ans),
            "note": "both arms: same within-document pool, same budget, same "
                    "prompt frame, same scorer; only the ranking differs",
            "summary": {a: {k: agg(a, k) for k in
                            ("answer_em", "osc", "tokens", "n_cells")}
                        for a in ("mt2net_bert", "ours")},
            "paired": {m: mcnemar([r["ours"][m] for r in ans],
                                  [r["mt2net_bert"][m] for r in ans])
                       for m in ("answer_em", "osc")},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    rec_path = str(Path(args.out).with_suffix("")) + "_records.jsonl"
    with open(rec_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    if ans:
        with open(str(Path(args.out).with_suffix("")) + "_answer.jsonl", "w") as fh:
            for rec in ans:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        al = out["answer_leg"]
        print(f"\n=== answer leg (budget {args.budget}, {llm.name}) ===")
        for a, s_ in al["summary"].items():
            print(f"  {a:12} EM={s_['answer_em']:.3f} OSC={s_['osc']:.3f} "
                  f"tok={s_['tokens']:.0f} cells={s_['n_cells']:.1f}")
        for m, v in al["paired"].items():
            print(f"  ours vs mt2net {m}: {v['only_first']}:{v['only_second']} "
                  f"p={v['exact_p']}")

    s = out["by_scheme"][args.scheme]["mt2net_bert"]
    print(f"\n=== mt2net_bert ({args.scheme}) ===")
    print(f"  hit@10={s['hit@10']:.3f} recall@10={s['recall@10']:.3f} mrr={s['mrr']:.3f} "
          f"ndcg@10={s['ndcg@10']:.3f} set_em@10={s['set_em@10']:.3f} "
          f"set_em@50={s['set_em@50']:.3f}")
    print(f"\nwrote -> {args.out}  (+ records -> {rec_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
