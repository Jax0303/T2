#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""AITQA column-selection benchmark — external validity for the cross-encoder column
resolver on a second hierarchical-table dataset (airline SEC filings).

AITQA gives no operand-cell labels, so OSC is not computable; but its answer *values*
let us recover the gold **column** by value-matching (85% of questions resolve to a
unique column). We then measure col-recall@k for lexical / bi-encoder / cross-encoder
column selectors, exactly as on HiTab (`col_select_bench.py`).

Run: PYTHONPATH=. python scripts/aitqa_col_bench.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_agent.query.operand_decomposer import Embedder

KS = (1, 2, 3)
_TOK = re.compile(r"[a-z0-9]+")


def norm_num(v):
    s = re.sub(r"[\$,%\s]", "", str(v)).strip()
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def lexical_rank(q, headers, k):
    qt = set(_TOK.findall(q.lower()))
    scored = []
    for i, h in enumerate(headers):
        ht = set(_TOK.findall(h.lower()))
        scored.append((len(qt & ht), -len(ht), i))  # overlap, prefer specific, stable
    scored.sort(reverse=True)
    return [i for _, _, i in scored[:k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/aitqa")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-encoder", default="BAAI/bge-reranker-base")
    ap.add_argument("--out", default="results/aitqa_col_bench.json")
    args = ap.parse_args()

    tabs = {}
    for line in open(f"{args.dir}/aitqa_tables.jsonl"):
        d = json.loads(line)
        tabs[d["id"]] = d
    qs = [json.loads(l) for l in open(f"{args.dir}/aitqa_questions.jsonl")]

    emb = Embedder(args.embed_model, device="cpu")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.cross_encoder)

    # build per-question (question, headers, gold_col) where gold col is unique
    items = []
    for q in qs:
        t = tabs.get(q["table_id"])
        if not t or not t.get("data"):
            continue
        headers = [" > ".join(h) if isinstance(h, list) else str(h)
                   for h in t["column_header"]]
        ans = {norm_num(a) for a in q["answers"] if norm_num(a) is not None}
        if not ans:
            continue
        cols = set()
        for row in t["data"]:
            for c, v in enumerate(row):
                if c < len(headers) and norm_num(v) in ans:
                    cols.add(c)
        if len(cols) == 1:  # unique gold column recovered
            items.append((q["question"], headers, next(iter(cols))))
    n = len(items)
    print(f"[pop] AITQA questions with a unique value-matched gold column: {n}")

    hits = {"lexical": {k: 0 for k in KS}, "embed": {k: 0 for k in KS},
            "cross": {k: 0 for k in KS}}
    per_q = {s: {k: [] for k in KS} for s in hits}   # per-query 0/1 for McNemar
    for question, headers, gold in items:
        lex = lexical_rank(question, headers, max(KS))
        qv = np.asarray(emb.encode([question])[0])
        hm = np.asarray(emb.encode(headers))
        emb_order = list(np.argsort(-(hm @ qv))[:max(KS)])
        sc = ce.predict([(question, h) for h in headers])
        cr_order = sorted(range(len(headers)), key=lambda i: -float(sc[i]))[:max(KS)]
        for k in KS:
            for s, order in (("lexical", lex), ("embed", emb_order), ("cross", cr_order)):
                ok = int(gold in order[:k])
                hits[s][k] += ok
                per_q[s][k].append(ok)

    # paired exact McNemar (binomial on discordant pairs), cross vs each baseline
    from scipy.stats import binomtest
    sig = {}
    for base in ("lexical", "embed"):
        for k in KS:
            b01 = sum(1 for c, b in zip(per_q["cross"][k], per_q[base][k]) if c and not b)
            b10 = sum(1 for c, b in zip(per_q["cross"][k], per_q[base][k]) if b and not c)
            p = float(binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else None
            sig[f"cross_vs_{base}@{k}"] = {"cross_only": b01, "base_only": b10,
                                           "p_two_sided": p}

    out = {"dataset": "AITQA", "population": {"unique_gold_col": n},
           "metric": "col_recall@k",
           "selectors": {s: {f"@{k}": round(hits[s][k] / n, 3) for k in KS} for s in hits},
           "mcnemar_exact": sig}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{'selector':<10}" + "".join(f"  col-recall@{k}" for k in KS))
    for s in hits:
        print(f"{s:<10}" + "".join(f"{out['selectors'][s][f'@{k}']:>14.3f}" for k in KS))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
