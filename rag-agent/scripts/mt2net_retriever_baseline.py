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
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = run_env(args.seed, args.model)

    total = args.eval_queries + args.train_queries
    queries, docs = load_population(total)
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
    per_query, records = {}, []
    for qi, q in enumerate(ev_pop):
        idxs = ev_pools[q["uid"]]
        gold = {int(g) for g in q["gold"]}
        scores = predict([(q["question"], ev_texts[gi]) for gi in idxs])
        rank_of = {}
        for pos_, local in enumerate(np.argsort(-np.asarray(scores)), 1):
            gi = idxs[int(local)]
            if gi in gold:
                rank_of[gi] = pos_
                if len(rank_of) == len(gold):
                    break
        per_query[qi] = [rank_of.get(g) for g in gold]
        for g in gold:
            records.append({"scheme": args.scheme, "retriever": "mt2net_bert",
                            "query": qi, "cell": g, "rank": rank_of.get(g)})

    out = {
        "env": env,
        "leg": "MT2Net's BERT-base binary-classifier retriever, trained here and "
               "evaluated on the same population/pool/gold/sentences as "
               "operand_collision_within_doc.py, so the scorer is the only variable",
        "population": {"name": "multihiertt_arith_multi_within_doc",
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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    rec_path = str(Path(args.out).with_suffix("")) + "_records.jsonl"
    with open(rec_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    s = out["by_scheme"][args.scheme]["mt2net_bert"]
    print(f"\n=== mt2net_bert ({args.scheme}) ===")
    print(f"  hit@10={s['hit@10']:.3f} recall@10={s['recall@10']:.3f} mrr={s['mrr']:.3f} "
          f"ndcg@10={s['ndcg@10']:.3f} set_em@10={s['set_em@10']:.3f} "
          f"set_em@50={s['set_em@50']:.3f}")
    print(f"\nwrote -> {args.out}  (+ records -> {rec_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
