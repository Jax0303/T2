#!/usr/bin/env python3
"""FinQA RAG: does retrieval completeness (not serialization) bottleneck numeric
answering, and does OPERAND-COMPLETE retrieval close the oracle gap?

FinQA gives, per question: a long context (pre_text + table + post_text), a gold
program (e.g. divide(60,243),multiply(#0,const_100)), the executed numeric answer
(exe_ans), and gold_inds = the EXACT evidence units (specific table rows / text
sentences) the computation needs — i.e. ground-truth operands. This lets us measure
operand-complete retrieval directly. FinQA is the core of T2-RAGBench.

Retrievable units = one per table row (header-attached sentence) + one per text
sentence. Retrieval modes:
  oracle          : all units (upper bound; no retrieval loss)
  operand_oracle  : exactly the gold_inds units (perfect operand-complete retrieval)
  bm25            : top-k units by BM25(question)               [baseline RAG]

The LLM reads the retrieved units and outputs the numeric answer; scored against
exe_ans with relative tolerance. seed=42, Groq LLM.

If bm25 << oracle and operand_oracle ~ oracle, then retrieval completeness is the
bottleneck and closing it closes the numeric gap -> motivates the method.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402
from rank_bm25 import BM25Okapi                                       # noqa: E402
from rag_agent.llm.factory import build_llm                           # noqa: E402
from codegen_flat_eval import complete_retry                          # noqa: E402

SEED = 42
FINQA = "/tmp/finqa"
_TOK = re.compile(r"[a-z0-9.]+")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def tok(s):
    return _TOK.findall(str(s).lower())


def row_sentence(table, r):
    """Header-attached serialization of a table row (FinQA gold_inds style)."""
    header = table[0]
    row = table[r]
    label = str(row[0]).strip() or f"row {r}"
    parts = []
    for c in range(1, len(row)):
        col = str(header[c]).strip() if c < len(header) else f"col{c}"
        parts.append(f"the {label} of {col} is {row[c]}")
    return " ; ".join(parts) + " ;"


def build_units(ex):
    """Return ordered list of (key, text) retrievable units, matching FinQA keys."""
    units = []
    for i, t in enumerate(ex.get("pre_text", [])):
        units.append((f"text_{i}", str(t)))
    table = ex.get("table") or []
    base = len(ex.get("pre_text", []))
    for r in range(1, len(table)):           # row 0 = header
        units.append((f"table_{r}", row_sentence(table, r)))
    for j, t in enumerate(ex.get("post_text", [])):
        units.append((f"text_{base + len(table) - 1 + j}", str(t)))
    return units


def retrieve(units, ex, question, mode, top_k):
    keys = [k for k, _ in units]
    texts = [t for _, t in units]
    gold_keys = set((ex["qa"].get("gold_inds") or {}).keys())
    if mode == "oracle":
        idx = list(range(len(units)))
    elif mode == "operand_oracle":
        idx = [i for i, k in enumerate(keys) if k in gold_keys] or list(range(min(top_k, len(units))))
    else:  # bm25
        bm = BM25Okapi([tok(t) for t in texts])
        idx = list(np.argsort(-bm.get_scores(tok(question)))[:top_k])
    got = bool(gold_keys) and gold_keys.issubset({keys[i] for i in idx})
    return [units[i] for i in sorted(idx)], got


ANSWER_SYS = (
    "You are a financial analyst. Using ONLY the provided evidence, compute the answer "
    "to the question. Do the arithmetic carefully. Output ONLY the final numeric value "
    "(a plain number; if it is a percentage give the number without the % sign).")


def build_prompt(units, question):
    ev = "\n".join(f"- {t}" for _, t in units)
    return f"Evidence:\n{ev}\n\nQuestion: {question}\n\nFinal numeric answer:"


def parse_num(txt):
    m = _NUM.findall(txt.replace("%", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def num_match(pred, gold, rtol=0.01, atol=0.1):
    if pred is None or gold is None:
        return False
    try:
        g = float(gold)
    except (ValueError, TypeError):
        return False
    return abs(pred - g) <= max(atol, rtol * abs(g))


def load_finqa(split, n):
    j = json.load(open(f"{FINQA}/{split}.json"))
    import random
    random.Random(SEED).shuffle(j)
    out = []
    for ex in j:
        qa = ex.get("qa", {})
        if qa.get("exe_ans") is None or not ex.get("table"):
            continue
        try:
            float(qa["exe_ans"])
        except (ValueError, TypeError):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--modes", default="oracle,operand_oracle,bm25")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--llm", default="groq:llama-3.1-8b-instant")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--out", default="results/codegen/finqa_rag.json")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    exs = load_finqa(args.split, args.n)
    llm = build_llm(args.llm)
    print(f"FinQA RAG | n={len(exs)} | modes={modes} | top_k={args.top_k} | {args.llm}", flush=True)

    per = {m: [] for m in modes}
    recall = {m: [] for m in modes}
    rows = []
    t0 = time.time()
    for i, ex in enumerate(exs, 1):
        units = build_units(ex)
        q = ex["qa"]["question"]
        gold = ex["qa"]["exe_ans"]
        rec = {"id": ex["id"], "q": q, "gold": gold, "n_units": len(units), "pred": {}, "ok": {}, "got": {}}
        for m in modes:
            ru, got = retrieve(units, ex, q, m, args.top_k)
            recall[m].append(int(got))
            try:
                pred = parse_num(complete_retry(llm, ANSWER_SYS, build_prompt(ru, q), args.max_tokens))
            except Exception as e:  # noqa: BLE001
                pred = None
                rec.setdefault("err", {})[m] = type(e).__name__
            ok = num_match(pred, gold)
            rec["pred"][m] = pred
            rec["ok"][m] = bool(ok)
            rec["got"][m] = bool(got)
            per[m].append(int(bool(ok)))
            time.sleep(args.sleep)
        rows.append(rec)
        if i % 10 == 0 or i == len(exs):
            acc = {m: np.mean(per[m]) for m in modes}
            print(f"  {i}/{len(exs)} " + " ".join(f"{m}={acc[m]:.2f}(R{np.mean(recall[m]):.2f})" for m in modes)
                  + f"  {time.time()-t0:.0f}s", flush=True)

    out = {"config": vars(args), "n": len(exs),
           "acc": {m: round(float(np.mean(per[m])), 4) for m in modes},
           "evidence_recall": {m: round(float(np.mean(recall[m])), 4) for m in modes},
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\nACC:", out["acc"])
    print("EVIDENCE RECALL:", out["evidence_recall"])
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
