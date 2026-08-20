# SPDX-License-Identifier: MIT
"""The head-to-head table, emitted from the records rather than retyped.

Every number here has been retyped by hand at least once already and drifted
(RESULTS.md 3e1de76). So it is generated: LaTeX for the paper, markdown for
the working docs, both off the same records files.

Ours is the header-path cell sentence, plus the table title WHERE THE CORPUS
HAS TITLES. That condition is not a per-dataset knob: it is decided by one
label-free corpus statistic, and the statistic is not close.

    HiTab        424 tables,  99.3% carry a title
    MultiHiertt 1310 tables,   0.0%
    AIT-QA       113 tables,   0.0%

    (scripts/, see TITLE_COVERAGE below; reproduce by counting non-empty
     Corpus.title over the same corpora the runs build)

The S3 sentence exists to state the title -- "In the table 'X', among A, the
value of B is 122." On a corpus with no titles it renders the same facts as S2
in 1.25x the tokens, so at a fixed budget it carries ~20% fewer cells and loses
on coverage alone. Reporting that as "the title hurts here" was wrong: the
title was never in the sentence, because there was no title to put in it.

The baseline arms never read --cell-scheme, so their records from the `_h2h_*`
runs are the same experiment at the same budget on the same frozen population
-- verified, not assumed: the `flat` arm's predictions are byte-identical
across the two files, so the pipeline is deterministic and the join is exact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manual_sentence_ceiling import mcnemar

BUDGETS = (256, 512)

# share of tables carrying a title, measured over the corpora the runs build
TITLE_COVERAGE = {"HiTab": 0.993, "MultiHiertt": 0.0, "AIT-QA": 0.0}
TITLED = 0.5          # above this, the sentence has a title to state, so S3

# (label, baseline records, S3 records). The S3 file is None where the corpus
# has no titles: there S3 is only S2 in more words, and the rule never picks it.
DATASETS = [
    ("HiTab", "results/corpus_dump_vs_cell_h2h_dense_%d_records.jsonl",
     "results/corpus_dump_vs_cell_s3_dense_%d_records.jsonl"),
    ("MultiHiertt", "results/corpus_dump_vs_cell_h2h_mh_dense_%d_records.jsonl",
     "results/corpus_dump_vs_cell_s3_mh_dense_%d_records.jsonl"),
    ("AIT-QA", "results/corpus_dump_vs_cell_h2h_aitqa_dense_%d_records.jsonl",
     "results/corpus_dump_vs_cell_s3_aitqa_dense_%d_records.jsonl"),
]

# (group, row label, arm). group boundaries become \midrule. "ours" is resolved
# per dataset by TITLE_COVERAGE, so it is one method, not one column each.
ROWS = [
    ("Table",  "whole-table dump",           "dump"),
    ("Table",  "cell-ranked table dump",     "cell2dump"),
    ("Row",    "row chunk",                  "row"),
    ("Cell",   "top-1 table only (cascade)", "cascade"),
    ("Cell",   "leaf label only",            "flat"),
    ("Ours",   "header-path cell sentence",  "ours"),
]


def load(path):
    p = Path(path)
    return {r["query_id"]: r for r in map(json.loads, p.open())} if p.exists() else None


def ours_series(ds, base, s3):
    """The method's per-question outcomes on one dataset/budget.

    S3 where the corpus has titles for the sentence to state, S2 where it does
    not. Both arms are named `cell` in their own records file; which file is
    read is the whole of the rule.
    """
    if TITLE_COVERAGE[ds] > TITLED and s3 is not None:
        return [s3[q]["cell"]["answer_em"] for q in sorted(s3)], "S3", sorted(s3)
    return [base[q]["cell"]["answer_em"] for q in sorted(base)], "S2", sorted(base)


def cell_values():
    """(dataset, budget) -> {row index: (em, p_vs_ours)}, plus n and which scheme."""
    out, meta = {}, {}
    for ds, base_f, s3_f in DATASETS:
        for b in BUDGETS:
            B, S = load(base_f % b), load(s3_f % b)
            if not B:
                out[(ds, b)] = None
                continue
            if S is not None:
                S = {q: r for q, r in S.items() if q in B}
            ours, scheme, ids = ours_series(ds, B, S)
            meta[(ds, b)] = (len(ids), scheme)
            col = {}
            for i, (_, _, arm) in enumerate(ROWS):
                if arm == "ours":
                    col[i] = (sum(ours) / len(ours), None)
                    continue
                v = [B[q][arm]["answer_em"] for q in ids]
                col[i] = (sum(v) / len(v), mcnemar(ours, v)["exact_p"])
            out[(ds, b)] = col
    return out, meta


def fmt(em, p, best):
    s = f"{em * 100:.1f}"
    if best:
        s = f"\\textbf{{{s}}}"
    if p is not None and p < 0.05:
        s += "$^{\\dagger}$"
    return s


def latex(vals, meta):
    cols = [(ds, b) for ds, _, _ in DATASETS for b in BUDGETS]
    best = {c: max(v[1][0] for v in vals[c].items()) for c in cols if vals[c]}
    L = [r"\begin{tabular}{ll" + "cc" * len(DATASETS) + "}", r"\toprule"]
    L.append("\\multirow{2}{*}{Unit} & \\multirow{2}{*}{Index text} & "
             + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ds} ($n{{=}}{meta[(ds, BUDGETS[0])][0]}$)}}"
                          for ds, _, _ in DATASETS) + r" \\")
    L.append(" & ".join(["", ""] + [str(b) for _ in DATASETS for b in BUDGETS]) + r" \\")
    L.append(r"\midrule")
    prev = None
    for i, (grp, label, _) in enumerate(ROWS):
        if prev is not None and grp != prev:
            L.append(r"\midrule")
        cells = []
        for c in cols:
            if not vals[c]:
                cells.append("--")
                continue
            em, p = vals[c][i]
            cells.append(fmt(em, p, abs(em - best[c]) < 1e-9))
        L.append(f"{grp if grp != prev else ''} & {label} & " + " & ".join(cells) + r" \\")
        prev = grp
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def markdown(vals, meta):
    cols = [(ds, b) for ds, _, _ in DATASETS for b in BUDGETS]
    best = {c: max(v[1][0] for v in vals[c].items()) for c in cols if vals[c]}
    head = "| Unit | Index text | " + " | ".join(f"{ds} {b}" for ds, b in cols) + " |"
    L = [head, "|" + "---|" * (2 + len(cols))]
    for i, (grp, label, _) in enumerate(ROWS):
        cells = []
        for c in cols:
            em, p = vals[c][i]
            s = f"{em * 100:.1f}"
            if abs(em - best[c]) < 1e-9:
                s = f"**{s}**"
            if p is not None and p < 0.05:
                s += "†"
            cells.append(s)
        L.append(f"| {grp} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="both", choices=["latex", "markdown", "both"])
    args = ap.parse_args()
    vals, meta = cell_values()
    if args.format in ("latex", "both"):
        print(latex(vals, meta))
    if args.format == "both":
        print()
    if args.format in ("markdown", "both"):
        print(markdown(vals, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
