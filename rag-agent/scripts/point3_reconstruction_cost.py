#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Point 3: how much of the S2 retrieval win is REAL structure vs RECONSTRUCTION noise?

My method (S2) prepends each cell's full header path. On genuinely raw tables the
path is RECONSTRUCTED from the grid, and the row axis is the weak one (~0.82
exact on HiTab real grids vs ~0.97 for columns). The objection: is S2's edge real, or an
artifact that evaporates once the tree it stands on is itself half-wrong?

This isolates it on HiTab, which ships BOTH the gold header tree AND the real
source grid (``data/hitab/data/tables/raw/*.json``, with ``texts``/``merged_regions``):

  flat      : baseline serialization (leaf labels only)
  S2_gold   : my method on the GOLD tree            (perfect structure, ceiling)
  S2_recon  : my method on the RECONSTRUCTED tree   (what raw data actually gets)

Same encoder, same within-table pool, same gold operand targets; only the header
path each cell carries differs. Retrieval metric = set-EM@k (ALL gold operand
cells of the query ranked <= k — completeness). The gap:
  * S2_gold - S2_recon = the price of reconstruction error.
  * S2_recon - flat    = what the method still buys on raw data despite that error.

LLM-free (retrieval only). Run: PYTHONPATH=. python scripts/point3_reconstruction_cost.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import numpy as np

from rag_agent.bench.hitab import load_queries
from rag_agent.reconstruct import reconstruct_col_paths, reconstruct_row_paths
from rag_agent.retrieve.encoders import default_encoder
from tree_reconstruct_hitab_raw import align, norm, tree_lines

KS = (10, 20, 50)


def cell_text(rp, cp, v, scheme):
    if scheme == "flat":
        lab = " ".join(x for x in ((rp[-1] if rp else ""), (cp[-1] if cp else "")) if x)
        return f"{lab}: {v}" if lab else str(v)
    path = " > ".join([*rp, *cp])          # S2 (gold or recon depending on rp/cp)
    return f"{path}: {v}" if path else str(v)


def shuffle_ancestors(paths: dict, seed: int):
    """Permute each path's ANCESTOR prefix across lines, keeping its leaf.

    The decisive control for "is the S2 gain structure, or just longer strings?".
    Leaf label, path length distribution and the table's ancestor vocabulary all
    survive; only the line -> ancestor ASSIGNMENT is destroyed. If S2 keeps its
    edge over flat under this, the edge was never the hierarchy.

    Returns ``(shuffled, n_changed, n_total)``.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(paths)
    prefixes = [list(paths[k][:-1]) for k in keys]
    order = rng.permutation(len(keys))
    out, changed = {}, 0
    for pos, k in enumerate(keys):
        leaf = paths[k][-1:]
        new = prefixes[int(order[pos])] + list(leaf)
        out[k] = new
        changed += int(new != list(paths[k]))
    return out, changed, len(keys)


def build_table_paths(raw, bt):
    """Return per-bt-data-cell gold & reconstructed (row_path, col_path), or None.

    Mirrors tree_reconstruct_hitab_raw.score_table's alignment: map bt data
    indices onto raw grid lines, reconstruct with the merge-aware front-end at
    the gold header boundary, and hand back paths aligned to bt (i, j).
    """
    texts = raw.get("texts") or []
    if not texts:
        return None
    cols_c, _ = tree_lines(raw.get("top_root") or {}, "top")
    rows_c, _ = tree_lines(raw.get("left_root") or {}, "left")
    if not cols_c or not rows_c:
        return None
    nhr, nhc = min(rows_c), min(cols_c)
    if nhr <= 0 or nhc <= 0:
        return None
    al = align(texts, rows_c, cols_c, nhr, nhc, bt)
    if al is None:
        return None
    rows_c, cols_c, _rate = al  # rows_c[i] = grid row of bt data row i; cols_c[j] likewise
    # Texts-only, not reconstruct_paths_with_merges: on HiTab dev the merge-aware
    # front-end is DOMINATED on both axes (col .943/.975, row .692/.820), so
    # consuming the span markup costs accuracy rather than buying it. Kept as an
    # ablation behind tree_reconstruct_hitab_raw.py --use-merges.
    rec_cols = reconstruct_col_paths(texts, nhr, nhc)
    rec_rows = reconstruct_row_paths(texts, nhr, nhc)

    gold_rp, gold_cp, rec_rp, rec_cp = {}, {}, {}, {}
    col_hit = col_tot = row_hit = row_tot = 0
    for j, c in enumerate(cols_c):
        gp = list(bt.col_path(j))
        idx = c - nhc
        rp = list(rec_cols[idx]) if 0 <= idx < len(rec_cols) else []
        gold_cp[j], rec_cp[j] = gp, rp
        col_tot += 1; col_hit += int(norm(gp) == norm(rp))
    for i, r in enumerate(rows_c):
        gp = list(bt.row_path(i))
        idx = r - nhr
        rp = list(rec_rows[idx]) if 0 <= idx < len(rec_rows) else []
        gold_rp[i], rec_rp[i] = gp, rp
        row_tot += 1; row_hit += int(norm(gp) == norm(rp))
    return {"gold_rp": gold_rp, "gold_cp": gold_cp, "rec_rp": rec_rp, "rec_cp": rec_cp,
            "n_r": bt.n_rows, "n_c": bt.n_cols,
            "recon": (col_hit, col_tot, row_hit, row_tot)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--raw-dir", default="data/hitab/data/tables/raw")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--min-operands", type=int, default=2,
                    help="keep queries with >= this many gold operand cells")
    ap.add_argument("--out", default="results/point3_reconstruction_cost.json")
    args = ap.parse_args()

    queries, tables = load_queries(args.data_dir, args.split)
    raw_dir = Path(args.raw_dir)

    # reconstruct paths for every table that has a raw grid and aligns
    paths, recon_acc = {}, [0, 0, 0, 0]
    for tid, bt in tables.items():
        p = raw_dir / f"{tid}.json"
        if not p.exists():
            continue
        try:
            raw = json.load(open(p))
        except Exception:
            continue
        pt = build_table_paths(raw, bt)
        if pt is None:
            continue
        paths[tid] = pt
        for i in range(4):
            recon_acc[i] += pt["recon"][i]
    ch, ct, rh, rt = recon_acc
    print(f"[recon] aligned tables: {len(paths)}  col_exact={ch/ct:.4f} ({ct}) "
          f"row_exact={rh/rt:.4f} ({rt})  [target ~.975/.820 confirms wiring]", flush=True)

    # keep queries whose table aligned and whose gold operands are all in range
    pop = []
    for q in queries:
        pt = paths.get(q.gold_table_id)
        if pt is None:
            continue
        ops = [(op.row, op.col) for op in q.gold_operands]
        if len(ops) < args.min_operands:
            continue
        if any(not (0 <= r < pt["n_r"] and 0 <= c < pt["n_c"]) for r, c in ops):
            continue
        pop.append((q, ops))
    print(f"[pop] queries kept: {len(pop)} (>= {args.min_operands} operands, aligned table)",
          flush=True)

    enc = default_encoder(model_name=args.embed_model)

    # global cell list per scheme -> one encode pass each
    schemes = ("flat", "S2_gold", "S2_recon", "S2_shuffled", "S2_gold_trunc")
    cell_key = []                       # (tid, i, j)
    cell_of = {}                        # (tid,i,j) -> global idx
    texts = {s: [] for s in schemes}
    shuf_changed = shuf_total = 0
    for si, (tid, pt) in enumerate(paths.items()):
        sr, ch_r, tot_r = shuffle_ancestors(pt["rec_rp"], seed=si)
        sc, ch_c, tot_c = shuffle_ancestors(pt["rec_cp"], seed=si + 100000)
        pt["shuf_rp"], pt["shuf_cp"] = sr, sc
        shuf_changed += ch_r + ch_c
        shuf_total += tot_r + tot_c
        # Gold, but cut to the reconstructed path's depth from the leaf end.
        # Reconstruction comes out SHALLOWER than gold (recon_audit_hitab.json:
        # 1,201 lines shallower vs 81 deeper) and the ancestors it drops carry
        # slightly LESS question vocabulary than gold's. If dropping them is why
        # S2_recon out-retrieves S2_gold, this arm reproduces the gain with
        # every kept segment still exactly correct.
        pt["trunc_rp"] = {i: (g[-len(pt["rec_rp"][i]):] if pt["rec_rp"][i] else g)
                          for i, g in pt["gold_rp"].items()}
        pt["trunc_cp"] = {j: (g[-len(pt["rec_cp"][j]):] if pt["rec_cp"][j] else g)
                          for j, g in pt["gold_cp"].items()}
    for tid, pt in paths.items():
        bt = tables[tid]
        for i in range(pt["n_r"]):
            for j in range(pt["n_c"]):
                cell_of[(tid, i, j)] = len(cell_key)
                cell_key.append((tid, i, j))
                v = bt.data[i][j]
                texts["flat"].append(cell_text(pt["gold_rp"][i], pt["gold_cp"][j], v, "flat"))
                texts["S2_gold"].append(cell_text(pt["gold_rp"][i], pt["gold_cp"][j], v, "S2"))
                texts["S2_recon"].append(cell_text(pt["rec_rp"][i], pt["rec_cp"][j], v, "S2"))
                texts["S2_shuffled"].append(cell_text(pt["shuf_rp"][i], pt["shuf_cp"][j], v, "S2"))
                texts["S2_gold_trunc"].append(cell_text(pt["trunc_rp"][i], pt["trunc_cp"][j], v, "S2"))
    print(f"[cells] {len(cell_key)} data cells across {len(paths)} tables", flush=True)
    print(f"[shuffle] {shuf_changed}/{shuf_total} lines got a different ancestor prefix "
          f"({shuf_changed/max(shuf_total,1):.1%}) — the rest had nothing to permute",
          flush=True)

    vecs = {s: np.asarray(enc.encode(texts[s])) for s in schemes}
    q_vecs = np.asarray(enc.encode([q.question for q, _ in pop]))

    # per-table global cell index lists (the within-table pool)
    pool_of = {}
    for gi, (tid, i, j) in enumerate(cell_key):
        pool_of.setdefault(tid, []).append(gi)

    hits = {s: {k: 0 for k in KS} for s in schemes}
    for qi, (q, ops) in enumerate(pop):
        tid = q.gold_table_id
        pool = pool_of[tid]
        gold_gis = {cell_of[(tid, r, c)] for r, c in ops}
        for s in schemes:
            sub = vecs[s][pool]
            order = np.argsort(-(sub @ q_vecs[qi]))
            rank_of = {}
            for pos, local in enumerate(order, 1):
                gi = pool[int(local)]
                if gi in gold_gis:
                    rank_of[gi] = pos
                    if len(rank_of) == len(gold_gis):
                        break
            worst = max((rank_of.get(gi, 10**9) for gi in gold_gis), default=10**9)
            for k in KS:
                hits[s][k] += int(worst <= k)

    n = len(pop)
    setem = {s: {f"set_em@{k}": round(hits[s][k] / n, 4) for k in KS} for s in schemes}
    out = {
        "population": {"dataset": "hitab", "split": args.split, "n_queries": n,
                       "min_operands": args.min_operands,
                       "aligned_tables": len(paths)},
        "reconstruction_accuracy": {"col_exact": round(ch / ct, 4),
                                    "row_exact": round(rh / rt, 4),
                                    "note": "texts-only front-end at gold boundary; "
                                            "cross-check vs tree_reconstruct_hitab_raw_merged"},
        "retriever": "dense (bge-small, within-table pool)",
        "metric": "set-EM@k (all gold operand cells ranked <= k)",
        "arms": {"flat": "baseline serialization (leaf labels only)",
                 "S2_gold": "my method on GOLD tree (perfect structure = ceiling)",
                 "S2_recon": "my method on RECONSTRUCTED tree (what raw data gets)",
                 "S2_shuffled": "CONTROL: reconstructed paths with the ancestor "
                                "prefix permuted across lines — same length, same "
                                "vocabulary, wrong assignment. S2_recon - S2_shuffled "
                                "is the part of the gain that is actually structural; "
                                "S2_shuffled - flat is the part that is just longer strings.",
                 "S2_gold_trunc": "CONTROL: GOLD paths cut to the reconstructed depth "
                                  "from the leaf end — every kept segment correct, but "
                                  "as short as reconstruction. Tests whether S2_recon "
                                  "out-retrieves S2_gold because gold's extra ancestors "
                                  "dilute the embedding rather than because recon is better."},
        "shuffle_control": {"lines_permuted": shuf_changed, "lines_total": shuf_total,
                            "rate": round(shuf_changed / max(shuf_total, 1), 4)},
        "set_em": setem,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)

    print("\n=== POINT 3: reconstruction cost (HiTab, dense, within-table) ===")
    print(f"n={n} queries | recon col={ch/ct:.3f} row={rh/rt:.3f}")
    for k in KS:
        f, g, r, sh, gt = (setem[s][f"set_em@{k}"] for s in schemes)
        print(f"  set-EM@{k:<2}  flat {f:.3f}  shuf {sh:.3f}  gold {g:.3f}  gold_trunc {gt:.3f}"
              f"  recon {r:.3f}   | gain={r-f:+.3f}"
              f"  (structural {r-sh:+.3f} / length-only {sh-f:+.3f})"
              f"  dilution={gt-g:+.3f}")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
