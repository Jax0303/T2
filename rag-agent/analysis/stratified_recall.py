#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 3 -- does header-path coverage predict whether the chunk gets retrieved?

Takes the chunking policies :mod:`analysis.header_path_coverage` measures, runs
the repo's retriever over the WHOLE corpus chunked that way, and splits each
gold cell by its own HPC verdict:

    full_coverage=True   the chunk holding it also holds every header ancestor
    full_coverage=False  at least one ancestor is missing from that chunk

then reports Recall@10 / Recall@50 inside each group.

Recall@k is per GOLD CELL: 1 if the chunk that contains that cell is among the
top-k chunks the retriever returned for its query. A cell that landed in no
chunk at all (a token window cut its field in half) can never be retrieved and
is reported as its own row rather than folded into either group.

The two groups are DISJOINT sets of cells, so a paired bootstrap is not defined
on them. The CI below is a two-sample percentile bootstrap: each group is
resampled with replacement independently, B and seed as specified, and the
statistic is the difference of the two means. The p-value is Fisher's exact
test on the 2x2 (group x hit) table, UNCORRECTED for multiple comparisons.

  PYTHONPATH=. .venv/bin/python analysis/stratified_recall.py \
      --dataset hitab --population hitab_dev_lookup_all --policy all
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import corpus_dump_vs_cell as cdv                                    # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from header_path_coverage import (chunks_for, header_path,           # noqa: E402
                                  load_corpus, norm)
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from rag_agent.serialization.base import Chunk                       # noqa: E402

# P3_whole_table is excluded by the spec: HPC_full = 1.0 on all four corpora,
# so the full_coverage=False group is empty and there is nothing to stratify.
POLICIES = ("P1_fixed_256", "P1_fixed_512", "P1_fixed_1024",
            "P2_row", "P4_path_cell")
KS = (10, 50)
MIN_N = 30


def build_chunks(C, policy, bud, by_table, scheme):
    """Every chunk of every table, in one flat list. The haystack is the corpus."""
    chunks, owner = [], []          # owner[n] = {(i, j)} cells inside chunk n
    for tid in C.tids:
        have = by_table.get(tid)
        if not have:
            continue                # no indexed cell (RealHiTBench blank padding)
        for ch in chunks_for(C, tid, policy, bud, have, scheme):
            chunks.append(Chunk(table_id=tid, chunk_id=ch["chunk_id"],
                                text=ch["text"], scheme=policy, kind="chunk"))
            owner.append((tid, ch["cells"]))
    return chunks, owner


def two_sample_bootstrap(a, b, B, seed):
    """Percentile CI on mean(a) - mean(b); groups resampled independently."""
    if not a or not b:
        return (None, None)
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = rng.integers(0, len(a), size=(B, len(a)))
    db = rng.integers(0, len(b), size=(B, len(b)))
    d = a[da].mean(1) - b[db].mean(1)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def fisher(a, b):
    """Uncorrected two-sided Fisher exact p on (group x hit)."""
    from scipy.stats import fisher_exact
    if not a or not b:
        return None
    t = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
    return float(fisher_exact(t)[1])


def rhb_multi_match(C):
    """RealHiTBench query ids whose answer resolves to >= 2 cells of the gold table."""
    def cn(s):
        return re.sub(r"[\s,$%]", "", str(s)).strip().lower()
    qa = json.load(open("data/realhitbench/QA_final.json"))["queries"]
    raw = {str(q["id"]): q for q in qa}
    by_tid = defaultdict(list)
    for k, (tid, i, j) in enumerate(C.cell_owner):
        by_tid[tid].append(k)
    val = {k: cn(C.cell_text[k].rsplit(": ", 1)[-1]) for k in range(len(C.cell_owner))}
    out = set()
    for q in C.queries:
        qid, tid = str(q["query_id"]), q["gold_table"]
        r = raw.get(qid)
        if r is None:
            continue
        s = str(r.get("ProcessedAnswer") or r.get("FinalAnswer") or "")
        want = {cn(x) for x in s.split(",") if cn(x)}
        if sum(1 for k in by_tid[tid] if val[k] in want) >= 2:
            out.add(qid)
    return out


def measure(C, pop, dataset, policy, bud, scheme, enc, alpha, B, seed, drop=frozenset()):
    by_table = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by_table[t][(i, j)] = n

    t0 = time.time()
    chunks, owner = build_chunks(C, policy, bud, by_table, scheme)
    of_chunk = {c.chunk_id: n for n, c in enumerate(chunks)}
    holds = defaultdict(list)                    # (tid, i, j) -> [chunk index]
    for n, (tid, cells) in enumerate(owner):
        for (i, j) in cells:
            holds[(tid, i, j)].append(n)
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    print(f"  [{policy}] {len(chunks)} chunks, index in {time.time() - t0:.0f}s",
          flush=True)

    def scores(question):
        bm = ix._bm25_scores(question)
        if alpha == 0.0:
            return bm
        dn = ix._dense_scores(question)
        if alpha == 1.0:
            return dn
        return alpha * _minmax(dn) + (1 - alpha) * _minmax(bm)

    kmax = max(KS)
    rows, t0 = [], time.time()
    for n_q, q in enumerate(pop, 1):
        if str(q["query_id"]) in drop:
            continue
        gold = [g for g in sorted(q["gold_cells"]) if g[0] in by_table
                and (g[1], g[2]) in by_table[g[0]]]
        if not gold:
            continue
        s = scores(q["question"])
        top = np.argpartition(-s, min(kmax, len(s) - 1))[:kmax]
        top = top[np.argsort(-s[top])]
        rank_of = {int(p): r for r, p in enumerate(top)}
        for (tid, i, j) in gold:
            rp, cp, v = C.cell_paths[by_table[tid][(i, j)]]
            path = header_path(rp, cp)
            if not path:
                continue                          # the cell IS a header; excluded
            cs = holds.get((tid, i, j), [])
            if not cs:
                grp, hits = "NOT_IN_ANY_CHUNK", {k: 0 for k in KS}
            else:
                # a cell sits in exactly one chunk under every policy here, but
                # take the best rank rather than assume it
                best = min(rank_of.get(c, 10 ** 9) for c in cs)
                # coverage is a property of the chunk that HOLDS the cell -- the
                # same chunk header_path_coverage scored. Under all five policies
                # here the chunks partition the table, so cs has one element;
                # the crosscheck against the Phase 2 CSV is what proves it.
                nt = norm(chunks[cs[0]].text)
                grp = "full" if all(norm(e) in nt for e in path) else "partial"
                hits = {k: int(best < k) for k in KS}
            rows.append({"query_id": str(q["query_id"]), "table_id": tid,
                         "gold_row": i, "gold_col": j, "group": grp, **hits})
        if n_q % 200 == 0:
            print(f"    {n_q}/{len(pop)}  {time.time() - t0:.0f}s", flush=True)

    out = {"dataset": dataset, "policy": policy, "n_chunks": len(chunks),
           "n_rows": len(rows), "groups": {}}
    for grp in ("full", "partial", "NOT_IN_ANY_CHUNK"):
        sel = [r for r in rows if r["group"] == grp]
        out["groups"][grp] = {"n": len(sel),
                              **{f"recall@{k}": (sum(r[k] for r in sel) / len(sel)
                                                 if sel else None) for k in KS}}
    for k in KS:
        a = [r[k] for r in rows if r["group"] == "full"]
        b = [r[k] for r in rows if r["group"] == "partial"]
        lo, hi = two_sample_bootstrap(a, b, B, seed)
        out[f"diff@{k}"] = {"delta": (sum(a) / len(a) - sum(b) / len(b))
                            if a and b else None,
                            "ci95_lo": lo, "ci95_hi": hi,
                            "p_uncorrected": fisher(a, b),
                            "n_full": len(a), "n_partial": len(b)}
    return out, rows


def verify_against_hpc(dataset, policy, rows, out_dir):
    """Cross-check the full/partial split against the Phase 2 CSV, cell by cell."""
    f = Path(out_dir) / f"{dataset}_{policy}_records.csv"
    if not f.exists():
        return "HPC CSV 없음"
    want = {}
    for r in csv.DictReader(open(f)):
        key = (r["query_id"], r["table_id"], int(r["gold_row"]), int(r["gold_col"]))
        want[key] = ("NOT_IN_ANY_CHUNK" if r["chunk_id"] == "NOT_IN_ANY_CHUNK"
                     else "full" if r["full_coverage"] == "True" else "partial")
    seen = {(r["query_id"], r["table_id"], r["gold_row"], r["gold_col"]): r["group"]
            for r in rows}
    common = set(want) & set(seen)
    bad = [k for k in common if want[k] != seen[k]]
    return (f"{len(common)}/{len(want)} 대조, 불일치 {len(bad)}"
            + (f" (예: {bad[:3]})" if bad else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hitab",
                    choices=["hitab", "multihiertt", "aitqa", "realhitbench"])
    ap.add_argument("--policy", default="all")
    ap.add_argument("--cell-scheme", default="S3c",
                    choices=["S2", "S3", "S3c", "mt2net"])
    ap.add_argument("--retriever", default="hybrid", choices=list(cdv.ALPHA))
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--rhb-question-types", nargs="*", default=[])
    ap.add_argument("--rhb-em-only", action="store_true")
    ap.add_argument("--rhb-drop-multimatch", action="store_true",
                    help="drop the queries whose answer resolves to >= 2 cells")
    ap.add_argument("--mh-queries", type=int, default=400)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--hpc-dir", default="results/hpc")
    ap.add_argument("--out-dir", default="results/hpc/stratified")
    a = ap.parse_args()

    pols = POLICIES if a.policy == "all" else tuple(
        p.strip() for p in a.policy.split(",") if p.strip())
    bad = set(pols) - set(POLICIES)
    if bad:
        ap.error(f"unknown policy {sorted(bad)}; pick from {POLICIES}")

    t0 = time.time()
    C = load_corpus(a)
    pop = C.queries[:a.max_queries] if a.max_queries else C.queries
    drop = frozenset()
    if a.rhb_drop_multimatch:
        if a.dataset != "realhitbench":
            ap.error("--rhb-drop-multimatch is realhitbench only")
        drop = frozenset(rhb_multi_match(C))
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_owner)} cells | "
          f"[pop] {len(pop)} queries (drop {len(drop)}) | {time.time() - t0:.0f}s",
          flush=True)

    bud = Budget(a.embed_model)
    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model),
                             a.cache_dir, f"{a.dataset}_{a.split}_{a.embed_model}")
    tag = a.dataset + ("_nomulti" if drop else "")
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in pols:
        res, rows = measure(C, pop, a.dataset, p, bud, a.cell_scheme, enc,
                            a.alpha, a.bootstrap, a.seed, drop)
        res["retriever"] = a.retriever
        res["alpha"] = a.alpha
        res["cell_scheme"] = a.cell_scheme
        res["population"] = a.population if a.dataset == "hitab" else a.dataset
        res["dropped_queries"] = len(drop)
        res["hpc_crosscheck"] = ("생략 (multimatch 제외 조건)" if drop else
                                 verify_against_hpc(a.dataset, p, rows, a.hpc_dir))
        (outdir / f"{tag}_{p}_stratified.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False))
        with open(outdir / f"{tag}_{p}_cells.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["query_id", "table_id", "gold_row", "gold_col", "group",
                        *[f"hit@{k}" for k in KS]])
            for r in rows:
                w.writerow([r["query_id"], r["table_id"], r["gold_row"],
                            r["gold_col"], r["group"], *[r[k] for k in KS]])
        print(f"  [{p}] {json.dumps(res['groups'], ensure_ascii=False)}")
        print(f"        crosscheck: {res['hpc_crosscheck']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
