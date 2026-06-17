#!/usr/bin/env python3
"""Improve the diagnosed bottleneck: push HPIR header-path decomposition past the
~0.67 ceiling (fuzzy 0.61, embedding 0.67). LLM-free, so Groq throttling is irrelevant.

Matchers compared (BOTH-axes header recall on HiTab dev, gold = entity_link keys):
  fuzzy      : OriginalTable._fuzzy_score ranking (current deterministic HPIR)
  embed      : bge-small cosine(question, header-path)
  hybrid     : UNION of fuzzy top-k and embed top-k (dedup) -> higher recall at a
               slightly larger retrieval budget (operand-complete retrieval can over-fetch)
  hybrid+exp : hybrid but the query is first expanded (drop operation cue-words via
               HPIR's extract_target_terms) so header tokens dominate the match
Reports ROW/COL/BOTH recall per matcher + the budget used.
"""
from __future__ import annotations
import argparse, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np                                                    # noqa: E402
from rag_agent.data.loader import load_samples, load_table           # noqa: E402
from rag_agent.stores.original_store import build_original_table     # noqa: E402
from rag_agent.query.header_path_resolver import extract_target_terms  # noqa: E402
from hpir_accuracy_gate import gold_headers, covered                 # noqa: E402

SEED = 42


def distinct(ot, axis):
    n = ot.n_cols if axis == "col" else ot.n_rows
    pof = ot.col_path if axis == "col" else ot.row_path
    seen, out = set(), []
    for i in range(n):
        p = pof(i); k = " > ".join(p)
        if p and k not in seen:
            seen.add(k); out.append(p)
    return out


def fuzzy_rank(ot, terms, axis, k):
    qs = " ".join(terms)
    scored = [(ot._fuzzy_score(qs, p), " > ".join(p), p) for p in distinct(ot, axis)]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, _, p in scored[:k]]


def embed_rank(model, q, paths, k):
    if not paths:
        return []
    pe = model.encode([" > ".join(p) for p in paths], normalize_embeddings=True, show_progress_bar=False)
    qe = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    order = np.argsort(-(pe @ qe))[:k]
    return [paths[i] for i in order]


def union(a, b, cap):
    out, seen = [], set()
    for p in a + b:
        key = " > ".join(p)
        if key not in seen:
            seen.add(key); out.append(p)
        if len(out) >= cap:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="data/hitab")
    ap.add_argument("--kr", type=int, default=4, help="row budget per matcher")
    ap.add_argument("--kc", type=int, default=5, help="col budget per matcher")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

    ss = load_samples(args.data, "dev"); random.Random(SEED).shuffle(ss)
    tcache = {}
    matchers = ["fuzzy", "embed", "hybrid", "hybrid+exp"]
    R = {m: {"row": [], "col": [], "both": []} for m in matchers}
    used = 0
    for s in ss:
        tid = s.get("table_id"); gr, gc = gold_headers(s.get("linked_cells"))
        if not tid or (not gr and not gc):
            continue
        if tid not in tcache:
            raw = load_table(tid, args.data); tcache[tid] = build_original_table(raw) if raw else None
        ot = tcache[tid]
        if ot is None:
            continue
        q = s["question"]; terms = extract_target_terms(q)
        rows_d, cols_d = distinct(ot, "row"), distinct(ot, "col")
        cand = {}
        f_r, f_c = fuzzy_rank(ot, terms, "row", args.kr), fuzzy_rank(ot, terms, "col", args.kc)
        e_r, e_c = embed_rank(model, q, rows_d, args.kr), embed_rank(model, q, cols_d, args.kc)
        ee_r, ee_c = embed_rank(model, " ".join(terms), rows_d, args.kr), embed_rank(model, " ".join(terms), cols_d, args.kc)
        cand["fuzzy"] = (f_r, f_c)
        cand["embed"] = (e_r, e_c)
        cand["hybrid"] = (union(f_r, e_r, args.kr + args.kr), union(f_c, e_c, args.kc + args.kc))
        cand["hybrid+exp"] = (union(f_r, ee_r, args.kr + args.kr), union(f_c, ee_c, args.kc + args.kc))
        for m in matchers:
            rp, cp = cand[m]
            rok = all(covered(g, rp) for g in gr) if gr else None
            cok = all(covered(g, cp) for g in gc) if gc else None
            if rok is not None: R[m]["row"].append(int(rok))
            if cok is not None: R[m]["col"].append(int(cok))
            R[m]["both"].append(int(all(x for x in (rok, cok) if x is not None)))
        used += 1
        if used >= args.n:
            break

    print(f"HPIR matcher improvement | n={used} | budget rows={args.kr}(x2 hybrid) cols={args.kc}(x2 hybrid)")
    print(f"{'matcher':12} {'ROW':>6} {'COL':>6} {'BOTH':>7}")
    for m in matchers:
        print(f"{m:12} {np.mean(R[m]['row']):6.3f} {np.mean(R[m]['col']):6.3f} {np.mean(R[m]['both']):7.3f}")


if __name__ == "__main__":
    main()
