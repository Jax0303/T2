#!/usr/bin/env python3
"""End-to-end answer extraction on the FLAT corpus (OpenWikiTable).

The full task the professor asked for, on simple tables:

    query  →  ① find the right table (BM25 retrieval)
           →  ② extract the answer from that table (free-tier LLM reader)
           →  score Exact Match against the gold answer

This mirrors the existing HiTab answer pipeline (codegen_eval.py / agent.py)
so the SIMPLE-table numbers here are directly comparable to the COMPLEX-table
numbers there — that is the simple→complex axis.

Retrieval (①) runs anywhere, no network. The reader (②) needs a free LLM:
  --llm groq:llama-3.3-70b-versatile        (needs GROQ_API_KEY + egress)
  --llm local:Qwen/Qwen2.5-7B-Instruct      (needs a local GPU)

Without --llm the script runs retrieval-only: it reports R@1 and writes the
exact prompts that WOULD be sent, so the pipeline can be validated offline.

Example (once api.groq.com is on the network allowlist):
  GROQ_API_KEY=... python rag-agent/scripts/prep_answer_eval.py \
      --llm groq:llama-3.1-8b-instant --n-queries 200 \
      --out rag-agent/results/prep/owt_answer_groq8b.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.eval.metrics import exact_match  # noqa: E402
from rag_agent.prep.conditions import from_openwikitable, serialize  # noqa: E402

# reuse the BM25 backend + loader from the retrieval harness
from prep_retrieval_eval import BM25Retriever, load_owt  # noqa: E402


READER_SYSTEM = (
    "You answer a question using ONLY the given table. "
    "Reply with a single line: 'Final answer: <answer>'. "
    "Copy the value verbatim from the table; do not explain."
)
READER_USER = "Table:\n{table}\n\nQuestion: {question}\nFinal answer:"

_FINAL_RE = re.compile(r"final answer\s*:?\s*(.+)", re.IGNORECASE)


def parse_answer(raw: str) -> str:
    """Pull the answer out of the reader's free text."""
    if not raw:
        return ""
    m = None
    for m in _FINAL_RE.finditer(raw):
        pass  # keep the last 'Final answer:' occurrence
    text = (m.group(1) if m else raw.strip().splitlines()[-1]).strip()
    return text.strip().strip(".").strip()


class BM25Index:
    """BM25 over the corpus, returning ranked table indices (not just gold)."""

    def __init__(self, texts):
        from rank_bm25 import BM25Okapi

        from prep_retrieval_eval import _tokenize

        self._tok = _tokenize
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def topk(self, query, k):
        import numpy as np

        scores = self.bm25.get_scores(self._tok(query))
        return list(np.argsort(scores)[::-1][:k])


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--condition", default="C1",
                   help="serialization for BOTH the index and the reader prompt")
    p.add_argument("--top-k", type=int, default=1,
                   help="reader reads the top-1 table; >1 concatenates")
    p.add_argument("--max-rows", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm", default=None,
                   help="reader spec; omit for retrieval-only (offline) mode")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir or base / "data" / "openwikitable")
    tables, queries = load_owt(data_dir, args.split)
    idx_by_id = {t.table_id: i for i, t in enumerate(tables)}
    queries = [(qid, q, g) for qid, q, g in queries if g in idx_by_id]
    if args.n_queries and len(queries) > args.n_queries:
        queries = random.Random(args.seed).sample(queries, args.n_queries)

    # gold answers live in the raw split — reload to attach them
    answers = {}
    import pandas as pd
    raw = pd.read_json(
        Path("/tmp/owt_src/Open_WikiTable/data") / f"{args.split}.json")
    if "answer" in raw.columns:
        for _, r in raw.iterrows():
            answers[r["question_id"]] = r["answer"]

    print(f"corpus {len(tables)} | queries {len(queries)} | "
          f"condition {args.condition} | llm {args.llm or '(retrieval-only)'}")

    texts = [serialize(t, args.condition, max_rows=args.max_rows) for t in tables]
    index = BM25Index(texts)

    llm = None
    if args.llm:
        from rag_agent.llm.factory import build_llm
        llm = build_llm(args.llm)

    rows = []
    n_r1 = n_em = n_scored = 0
    for qid, question, gold in queries:
        ranked = index.topk(question, max(args.top_k, 1))
        gold_idx = idx_by_id[gold]
        r1 = int(ranked[0] == gold_idx)
        n_r1 += r1

        context = "\n\n---\n\n".join(
            serialize(tables[i], args.condition, max_rows=args.max_rows)
            for i in ranked[: args.top_k])

        rec = {"question_id": qid, "question": question, "gold_table": gold,
               "retrieved_table": tables[ranked[0]].table_id, "r1": r1}

        if llm is not None and qid in answers:
            raw_out = llm.complete(READER_SYSTEM,
                                   READER_USER.format(table=context, question=question),
                                   max_tokens=args.max_tokens)
            pred = parse_answer(raw_out)
            em = int(exact_match(pred, answers[qid]))
            n_em += em
            n_scored += 1
            rec.update({"gold_answer": answers[qid], "pred": pred,
                        "reader_raw": raw_out, "em": em})
        elif qid in answers:
            rec["gold_answer"] = answers[qid]
            rec["reader_prompt"] = READER_USER.format(table=context, question=question)
        rows.append(rec)

    n = len(queries)
    summary = {
        "n_queries": n,
        "R@1": round(n_r1 / n, 4) if n else 0.0,
        "EM": round(n_em / n_scored, 4) if n_scored else None,
        "n_scored": n_scored,
        "answer_EM_given_correct_table": None,
    }
    # EM conditional on retrieving the right table — isolates reader skill
    if n_scored:
        correct = [r for r in rows if r.get("em") is not None and r["r1"] == 1]
        if correct:
            summary["answer_EM_given_correct_table"] = round(
                sum(r["em"] for r in correct) / len(correct), 4)

    print(f"R@1={summary['R@1']}  EM={summary['EM']}  "
          f"EM|correct-table={summary['answer_EM_given_correct_table']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "rows": rows},
                  f, indent=1, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
