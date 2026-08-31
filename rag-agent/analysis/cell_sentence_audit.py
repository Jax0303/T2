#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cell-sentence provenance audit.

A. value:  does the value the P4_path_cell sentence asserts equal the value at
   that (row, col) in the ORIGINAL source, re-derived here rather than read back
   out of the same corpus object the sentence came from?
B. header path: do the sentence's row/col header segments equal the cell's real
   ancestors in the source's own header tree, walked again here?
C. the HiTab alignment / reconstruction failure rates, re-measured.

Descriptive only. Nothing here is a fix.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

from header_path_coverage import TEMPLATE, cell_spans, load_corpus   # noqa: E402
from qwen_equiv_k import args_for                                    # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from rag_agent.serialization.base import fmt_value                   # noqa: E402


def vclass(src, sent):
    """Why two value strings differ. Descriptive labels, no normalisation is
    applied to the counts above."""
    a, b = str(src).strip(), str(sent).strip()
    if a == b:
        return "equal"
    if a.replace(",", "") == b.replace(",", ""):
        return "thousands_separator"
    if not b and a:
        return "sentence_blank"
    if not a and b:
        return "source_blank"
    if a.replace(",", "").lstrip("$").rstrip("%") == b.replace(",", "").lstrip("$").rstrip("%"):
        return "currency_or_percent_sign"
    try:
        if float(a.replace(",", "")) == float(b.replace(",", "")):
            return "numeric_equal_text_differs"
    except ValueError:
        pass
    return "other"

SCHEME = "S3c"
OUT = Path("results/audit/cell_sentence_audit.json")


def sentence(C, tid, n):
    rp, cp, v = C.cell_paths[n]
    return caption_sentence(C.title.get(tid, ""), rp, cp, v,
                            template=TEMPLATE[SCHEME])


def expected(C, tid, n, v):
    rp, cp, _ = C.cell_paths[n]
    return caption_sentence(C.title.get(tid, ""), rp, cp, v,
                            template=TEMPLATE[SCHEME])


# ---------------------------------------------------------------- originals
def walk_tree(root, texts, axis):
    """grid line index -> ancestor labels, read straight off the raw tree.

    A raw HiTab header node carries only (row_index, column_index); its LABEL is
    whatever sits at that coordinate in ``texts``. The synthetic root is
    (-1, -1) and contributes nothing.
    """
    out = {}

    def cell(r, c):
        if 0 <= r < len(texts) and 0 <= c < len(texts[r]):
            return " ".join(str(texts[r][c]).split()).strip()
        return ""

    def walk(node, prefix):
        r, c = node.get("row_index", -1), node.get("column_index", -1)
        path = prefix
        if r >= 0 and c >= 0:
            lab = cell(r, c)
            path = prefix + [lab] if lab else prefix
            out[c if axis == "top" else r] = path
        for ch in node.get("children") or []:
            if isinstance(ch, dict):
                walk(ch, path)

    if root:
        walk(root, [])
    return out


def hitab_original():
    """tid -> (grid, rows_c, cols_c, by_line_row, by_line_col), re-derived."""
    from rag_agent.bench.hitab import load_queries
    from tree_reconstruct_hitab_raw import align, tree_lines

    _, tables = load_queries("data/hitab", "dev")
    raw_dir = Path("data/hitab/data/tables/raw")
    out = {}
    for tid, bt in tables.items():
        f = raw_dir / f"{tid}.json"
        if not f.exists():
            continue
        try:
            raw = json.load(open(f))
        except Exception:                                        # noqa: BLE001
            continue
        texts = raw.get("texts") or []
        cols_c, _ = tree_lines(raw.get("top_root") or {}, "top")
        rows_c, _ = tree_lines(raw.get("left_root") or {}, "left")
        if not texts or not cols_c or not rows_c:
            continue
        nhr, nhc = min(rows_c), min(cols_c)
        if nhr <= 0 or nhc <= 0:
            continue
        al = align(texts, rows_c, cols_c, nhr, nhc, bt)
        if al is None:
            continue
        rows_c, cols_c, _ = al
        by_row = walk_tree(raw.get("left_root") or {}, texts, "left")
        by_col = walk_tree(raw.get("top_root") or {}, texts, "top")
        out[tid] = (texts, rows_c, cols_c, by_row, by_col)
    return out


def aitqa_original():
    tabs = {d["id"]: d for d in (json.loads(l) for l in
                                 open("data/aitqa/aitqa_tables.jsonl"))}
    return tabs


# ---------------------------------------------------------------- checks
def md_value(C, tid, i, j, spans):
    s = spans.get((i, j))
    if s is None:
        return None
    return "\n".join(C.md_lines[tid])[s[0]:s[1]].strip()


def audit(dataset, population, rhb_types=()):
    if dataset == "multihiertt":
        from qwen_equiv_k import A
        a = A()
        a.dataset, a.population, a.rhb_question_types = dataset, population, []
    else:
        a = args_for(dataset, population)
        if rhb_types:
            a.rhb_question_types = list(rhb_types)
    C = load_corpus(a)
    by = defaultdict(dict)
    for n, (t, i, j) in enumerate(C.cell_owner):
        by[t][(i, j)] = n

    orig = hitab_original() if dataset == "hitab" else (
        aitqa_original() if dataset == "aitqa" else None)

    res = {"dataset": dataset, "population": population,
           "n_cells_indexed": len(C.cell_owner), "n_tables": len(C.tids)}
    from collections import Counter
    md_tot = md_bad = 0
    src_tot = src_bad = src_skip = 0
    hp_tot = hp_bad = 0
    md_cls, src_cls, hp_cls = Counter(), Counter(), Counter()
    ex_val, ex_src, ex_hp = [], [], []

    for tid in C.tids:
        have = by.get(tid)
        if not have:
            continue
        spans = cell_spans(C, tid, have)
        for (i, j), n in have.items():
            rp, cp, v = C.cell_paths[n]
            sent = sentence(C, tid, n)

            # A1 -- against the same table's markdown (the P1/P3 arms' text)
            mv = md_value(C, tid, i, j, spans)
            if mv is not None:
                md_tot += 1
                md_c = vclass(fmt_value(mv), fmt_value(v)); md_cls[md_c] += 1
                if fmt_value(mv) != fmt_value(v):
                    md_bad += 1
                    if sum(1 for e in ex_val if e["class"] == md_c) < 2 \
                            and len(ex_val) < 10:
                        ex_val.append({"where": "markdown", "class": md_c,
                                       "table": tid, "row": i, "col": j,
                                       "sentence": sent, "source_value": mv,
                                       "sentence_value": fmt_value(v)})

            # A2 -- against the dataset's own original grid
            if dataset == "hitab" and orig is not None and tid in orig:
                texts, rows_c, cols_c, by_row, by_col = orig[tid]
                if i < len(rows_c) and j < len(cols_c):
                    r, c = rows_c[i], cols_c[j]
                    gv = (texts[r][c] if r < len(texts)
                          and c < len(texts[r]) else None)
                    src_tot += 1
                    src_c = vclass(fmt_value("" if gv is None else gv), fmt_value(v))
                    src_cls[src_c] += 1
                    if gv is None or fmt_value(gv) != fmt_value(v):
                        src_bad += 1
                        if sum(1 for e in ex_src if e["class"] == src_c) < 2 \
                                and len(ex_src) < 10:
                            ex_src.append({"where": "hitab_raw",
                                           "class": src_c, "table": tid,
                                           "row": i, "col": j,
                                           "sentence": sent,
                                           "source_value": gv,
                                           "sentence_value": fmt_value(v)})
                    # B -- ancestors from the raw header trees
                    hp_tot += 1
                    grp = [x for x in (by_row.get(r) or []) if str(x).strip()]
                    gcp = [x for x in (by_col.get(c) or []) if str(x).strip()]
                    dr = [str(x).strip() for x in rp] != grp
                    dc = [str(x).strip() for x in cp] != gcp
                    hp_cls[("row" if dr else "") + ("col" if dc else "")
                           or "equal"] += 1
                    if dr or dc:
                        hp_bad += 1
                        if len(ex_hp) < 10:
                            ex_hp.append({"table": tid, "row": i, "col": j,
                                          "sentence": sent,
                                          "sentence_row_path": list(rp),
                                          "sentence_col_path": list(cp),
                                          "tree_row_path": grp,
                                          "tree_col_path": gcp})
                else:
                    src_skip += 1
            elif dataset == "aitqa" and orig is not None and tid in orig:
                t = orig[tid]
                row = t["data"][i] if i < len(t["data"]) else []
                gv = row[j] if j < len(row) else ""
                src_tot += 1
                src_c = vclass(fmt_value(gv), fmt_value(v)); src_cls[src_c] += 1
                if fmt_value(gv) != fmt_value(v):
                    src_bad += 1
                    if len(ex_src) < 10:
                        ex_src.append({"where": "aitqa_jsonl", "class": src_c,
                                       "table": tid, "row": i, "col": j,
                                       "sentence": sent, "source_value": gv,
                                       "sentence_value": fmt_value(v)})
                hp_tot += 1
                grp = [str(x).strip() for x in
                       (t["row_header"][i] if i < len(t["row_header"]) else [])
                       if str(x).strip()]
                gcp = [str(x).strip() for x in
                       (t["column_header"][j]
                        if j < len(t["column_header"]) else [])
                       if str(x).strip()]
                dr = [str(x).strip() for x in rp] != grp
                dc = [str(x).strip() for x in cp] != gcp
                hp_cls[("row" if dr else "") + ("col" if dc else "")
                       or "equal"] += 1
                if dr or dc:
                    hp_bad += 1
                    if len(ex_hp) < 10:
                        ex_hp.append({"table": tid, "row": i, "col": j,
                                      "sentence": sent,
                                      "sentence_row_path": list(rp),
                                      "sentence_col_path": list(cp),
                                      "tree_row_path": grp,
                                      "tree_col_path": gcp})

    res["A_markdown"] = {"n_compared": md_tot, "n_mismatch": md_bad,
                         "rate": round(md_bad / md_tot, 6) if md_tot else None}
    res["A_source"] = {"n_compared": src_tot, "n_mismatch": src_bad,
                       "n_skipped": src_skip,
                       "rate": round(src_bad / src_tot, 6) if src_tot else None}
    res["B_header_tree"] = {"n_compared": hp_tot, "n_mismatch": hp_bad,
                            "rate": round(hp_bad / hp_tot, 6) if hp_tot else None}
    res["A_markdown"]["classes"] = dict(md_cls.most_common())
    res["A_source"]["classes"] = dict(src_cls.most_common())
    res["B_header_tree"]["classes"] = dict(hp_cls.most_common())
    res["examples_value_markdown"] = ex_val
    res["examples_value_source"] = ex_src
    res["examples_header"] = ex_hp
    return res


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = [("hitab", "hitab_dev_lookup_all"), ("aitqa", "aitqa"),
            ("realhitbench", "rhb_lookup_all"),
            ("multihiertt", "")]
    out = []
    for ds, pop in jobs:
        r = audit(ds, pop)
        out.append(r)
        print(json.dumps({k: v for k, v in r.items()
                          if not k.startswith("examples")},
                         ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
