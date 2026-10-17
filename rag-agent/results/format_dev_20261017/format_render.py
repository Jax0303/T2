"""전달 형식 렌더링 — build_formats.py(커밋 ac9aea7, 고정)의 G·S·D 렌더링을 그대로 옮긴 모듈(882 확인용). 동작 동일성은
build_formats_882.py --verify-dev 가 dev D 레코드를 다시 만들어 커밋본과 글자 단위로 대조해 확인한다."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population          # noqa: E402
from retrieval_accuracy import build_corpus                          # noqa: E402

TBL = re.compile(r"^Table \d+$")
esc = lambda v: str(v).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", "")
strip_tbl = lambda path: [p for k, p in enumerate(path) if not (k == 0 and TBL.match(str(p)))]


def load_split(split, uids):
    _q, docs, _ = load_population(split)
    tables, hdr = build_tables({u: docs[u] for u in uids}, "v3.3u", "none")
    texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
    sent, coord = defaultdict(dict), defaultdict(dict)
    for txt, cov, tid in zip(texts, covers, unit_tids):
        c = next(iter(cov)); s = cell_id(c, hdr).split("::", 1)[1]
        sent[tid.split("::")[0]][s] = txt; coord[tid.split("::")[0]][s] = c
    return tables, hdr, sent, coord


def make_renderers(tables, hdr, S_, K_):
    short = lambda tid, i, j: cell_id((tid, i, j), hdr).split("::", 1)[1]
    def grouped(sel_ids):
        order, per = [], {}
        for s in sel_ids:
            tid, i, j = K_[s]
            if tid not in per: order.append(tid); per[tid] = []
            per[tid].append((i, j, s))
        blocks = []
        for tid in order:
            k = int(tid.split("::")[1]) + 1
            blocks.append(f"Table {k}\n" + "\n".join(S_[s] for i, j, s in sorted(per[tid])))
        return "\n\n".join(blocks), set(sel_ids)
    def subtable(sel_ids, dense):
        order, per = [], {}
        for s in sel_ids:
            tid, i, j = K_[s]
            if tid not in per: order.append(tid); per[tid] = {"rows": set(), "cols": set(), "cells": set()}
            per[tid]["rows"].add(i); per[tid]["cols"].add(j); per[tid]["cells"].add((i, j))
        blocks, delivered = [], set()
        for tid in order:
            t = tables[tid].table; rws = sorted(per[tid]["rows"]); cls = sorted(per[tid]["cols"]); k = int(tid.split("::")[1]) + 1
            lines = [f"Table {k}", "| | " + " | ".join(esc(" > ".join(strip_tbl(t.col_path(j)))) for j in cls) + " |", "| --- | " + " | ".join("---" for _ in cls) + " |"]
            for i in rws:
                vals = []
                for j in cls:
                    v = t.data[i][j]; show = (i, j) in per[tid]["cells"] or (dense and str(v).strip() != "")
                    vals.append(esc(v) if show else "")
                    if show and str(v).strip(): delivered.add(short(tid, i, j))
                lines.append("| " + esc(" > ".join(strip_tbl(t.row_path(i)))) + " | " + " | ".join(vals) + " |")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks), delivered
    return grouped, subtable


def select_render(rank, fn, T, n_tok):
    sel, best = [], ("", set())
    for u in rank:
        text, deliv = fn(sel + [u])
        if n_tok(text) > T: break
        sel.append(u); best = (text, deliv)
    return best[0], best[1], len(sel)
