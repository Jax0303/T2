# SPDX-License-Identifier: MIT
"""The cell->sentence templates, and nothing else.

Replaces the ``short``/``medium``/``long`` preset axis. Sentence length is no
longer an experimental variable; what remains is the one contrast that carries a
claim:

* :data:`MT2NET` — a historical, MT2Net-inspired serialization control. Zhao et al. (2022), MultiHiertt /
  MT2Net, ACL, arXiv:2206.01347 §4 renders each cell as a sentence carrying its
  hierarchical row and column headers, and retrieves the top-n such sentences.
  This template exists so "what the 2022 baseline indexes" is a thing this repo
  can actually run, rather than a thing it paraphrases.

* :data:`STRUCTURAL` — this work's index unit. Byte-identical to what the old
  ``length="long"`` preset produced, so every result already on disk that was
  produced under ``"long"`` stays reproducible from this code — with ONE
  exception, the case-fold fix in :func:`render`. Untitled-corpus results dated
  2026-08-23 or earlier (RealHiTBench, MultiHiertt, AIT-QA) were produced from
  case-folded sentences and reproduce only from the commit before that fix.
  HiTab results are unaffected: its labels are lower-case already.

* :data:`STRUCTURAL_COMPACT` — :data:`STRUCTURAL` where a title exists, the bare
  S2 path string where it does not. Not a length knob returning through the back
  door: the frame is the title's grammatical seat, so with no title it asserts
  nothing while still spending budget. Measured across three corpora, the sign of
  STRUCTURAL's advantage over MT2Net tracks title coverage — HiTab 99.3% titled
  gives +.082, RealHiTBench 38.2% gives +.004, AIT-QA 0% gives **-.062**
  (Holm-corrected, results/h2h830_*.json and results/h2h_untitled_*.json). This
  template is the mechanism's own prescription, pre-registered in
  PREREG-2026-08-23-compact-untitled.md before it was run.

* :data:`STRUCTURAL_LEAF` — :data:`STRUCTURAL_COMPACT` with the row/col LEAF
  labels (the last segment of each path — the token that actually varies
  between two cells that share a table and one axis) repeated as a prefix:
  ``"{row_leaf} / {col_leaf}: {structural_compact body}"``. Targets same-table,
  one-axis-right top-1 errors (results/bottleneck_root_cause/BOTTLENECK_ROOT_CAUSE.md,
  wrong_row+wrong_column = 61.4% of top-1 errors, 2026-09-16): two cells in the
  same table differ from their sentence's boilerplate only in a few leaf
  tokens buried mid-sentence, and repeating them was expected to weight them
  more heavily in a mean-pooled embedding. That assumption was WRONG: the
  deployed encoder (BAAI/bge-base-en-v1.5 via sentence-transformers) pools by
  CLS token, not mean (results/embedding_fusion_diagnosis_20260917/, read off
  the model's own ``1_Pooling`` config, ``pooling_mode: 'cls'`` — not
  templates.py's assumption, which was never checked against the model until
  that script did). A dense-only/sparse-only 2x2 (results/embedding_fusion_diagnosis_20260917/
  leaf_channel_ablation.json) decomposed which side of the hybrid score
  actually carries the gain: R@1 plain/plain .5752, leaf-on-dense-only .6135
  (+.0383), leaf-on-sparse-only .5792 (+.0040), both .6176 (+.0424) — the two
  channel contributions are almost exactly additive (.0383+.0040=.0423≈.0424,
  no interaction), and 90% of the gain is the DENSE side despite CLS (not
  mean) pooling: the repeated tokens must be shifting the CLS vector through
  self-attention, not through mean-pool averaging or BM25 term frequency. Not
  a length knob returning through the back door either way: it repeats
  content already in the sentence, it does not add new information.

* :data:`STRUCTURAL_LEAF_X2` — :data:`STRUCTURAL_LEAF` with its own leaf prefix
  repeated once more: ``"{row_leaf} / {col_leaf}: {structural_leaf body}"``.
  One repeat (STRUCTURAL_LEAF) moved R@1 .5752->.6176 and cut wrong_row
  113->91, wrong_column 143->126 of 991 (results/sentence_disambiguation_20260916/
  SENTENCE_DISAMBIGUATION.md) but left most of both error classes standing.
  This tested whether a second repeat of the SAME leaf tokens kept buying
  R@1/lowering wrong_row+wrong_column further, or had already plateaued.
  Result: plateaued and reversed slightly — R@1 .6176->.6145, wrong_row
  91->98, wrong_column 126->125 (results/sentence_disambiguation_20260916/
  SENTENCE_DISAMBIGUATION_structural_leaf_x2.md, 2026-09-17). Repetition count
  is not a lever worth pushing further on this template.

PROVISIONAL — CONFIRMED NOT TO MATCH (checked against the paper 2026-09-08).
The knobs below were reverse-engineered from the paper's rendered example, and
rendering that example's own cell with them does not reproduce it:

    ours  For Product of Innovation Systems of Segment, Sales, 2018, Year Ended December 31 is 2,894
    paper For Innovation Systems of Segment, sales of product in 2018, Year Ended December 31 is 2,894

The paper puts the ROW's leaf inside the COLUMN clause ("sales of product in
2018") and joins the year with " in "; these knobs concatenate the two axes with
fixed separators. The rendering is semantic, not a separator join.

Worse, the paper shows a SECOND rendered fact in Figure 1's "Retrieved top-n
Facts" -- "The funded Aerospace Systems in 2017 was 9560" -- whose shape ("The
... was X") disagrees with the first ("For ... is X"). Two examples, two shapes,
no stated rule.

What the paper DOES state (Section 4, verbatim) is the principle, not the form:

    "we turn each cell into a sentence, along with its hierarchical row and
     column headers"

This template shares the cell-plus-header idea, but is not a reproduction of
the official table_description strings, learned retriever, or reasoning modules.
The official 45bd9cc release reads precomputed table_description strings and uses
a RoBERTa-base pair classifier (inference_configs/retriever_inference.yaml). That
distinction is affordable because this repo already measured that the sentence's
FORM does not carry the gain -- S2 .517 / S2r .527 / MT2Net .516 are
indistinguishable (CLAUDE.md section 5) -- what carries it is the table's own
label, which MT2Net's unit does not have.

The paper's rendered example, for the record:

    "For Innovation Systems of Segment, sales of product in 2018,
     Year Ended December 31 is 2,894"

One example does not determine the rule. The knobs below are the readings that
match that string most literally; each is a judgement call awaiting confirmation,
and until they are confirmed the ``MT2NET`` label overstates what is verified.
Do not cite a number produced under this template as "MT2Net reproduction"
without saying which readings were used.

  MT2NET_ROW_SEP / MT2NET_COL_SEP
      The example joins the row path with " of " and the column path with ", ".
      Whether those are the general separators, or artefacts of these particular
      headers, is not stated.
  MT2NET_LEAF_FIRST
      "Innovation Systems of Segment" reads leaf-then-parent. This repo's own
      paths run root-to-leaf. Whether MT2Net emits leaf-first on both axes, or
      whether "Segment" is the stub column's *name* rather than a parent header
      level, is not stated.
  MT2NET_TRAILING_PERIOD
      The quoted example carries no terminal period.
"""
from __future__ import annotations

from typing import Sequence

from .base import fmt_value, join_path

MT2NET = "mt2net"
STRUCTURAL = "structural"
STRUCTURAL_COMPACT = "structural_compact"
STRUCTURAL_LEAF = "structural_leaf"
STRUCTURAL_LEAF_X2 = "structural_leaf_x2"
TEMPLATES = (MT2NET, STRUCTURAL, STRUCTURAL_COMPACT, STRUCTURAL_LEAF, STRUCTURAL_LEAF_X2)

# --- provisional readings of the single published MT2Net example ---
MT2NET_ROW_SEP = " of "
MT2NET_COL_SEP = ", "
MT2NET_LEAF_FIRST = True
MT2NET_TRAILING_PERIOD = False


def _path(path: Sequence[str], sep: str, leaf_first: bool) -> str:
    segs = list(path)
    if leaf_first:
        segs = segs[::-1]
    return join_path(segs, sep=sep)


def render(template: str, title, row_path: Sequence[str], col_path: Sequence[str],
           value=None) -> str:
    """Render one index-unit sentence.

    ``value=None`` omits the ``"is {value}"`` predicate, naming a header *scope*
    rather than asserting a cell's contents — used where the ranked thing is a
    header node, not an index unit. The deployed index always renders a value.
    """
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
    has_val = value is not None
    val_s = fmt_value(value) if has_val else ""
    title_s = fmt_value(title) if title else ""

    if template in (STRUCTURAL_LEAF, STRUCTURAL_LEAF_X2):
        inner = STRUCTURAL_COMPACT if template == STRUCTURAL_LEAF else STRUCTURAL_LEAF
        body = render(inner, title, row_path, col_path, value)
        row_leaf = fmt_value(row_path[-1]) if row_path else ""
        col_leaf = fmt_value(col_path[-1]) if col_path else ""
        prefix = " / ".join(p for p in (row_leaf, col_leaf) if p)
        return f"{prefix}: {body}" if prefix else body

    if template == STRUCTURAL_COMPACT and not title_s:
        # The frame ("in the table X, among ..., the value of ... is ...") exists
        # to seat a TITLE in a grammatical sentence. With no title it states
        # nothing and still costs tokens, and the budget pays in cells that fit:
        # on AIT-QA (0% titled) STRUCTURAL runs 24.3 tok/cell against MT2Net's
        # 20.3, a 1.20x capacity deficit that lands as a 1.21x OSC deficit and
        # -.062 answer EM (results/h2h_untitled_*.json). Fall back to the S2 path
        # string, byte-identical to point3_reconstruction_cost.cell_text(...,"S2").
        path = join_path([*row_path, *col_path])
        if not has_val:
            return path
        return f"{path}: {val_s}" if path else val_s

    if template == MT2NET:
        row_s = _path(row_path, MT2NET_ROW_SEP, MT2NET_LEAF_FIRST)
        col_s = _path(col_path, MT2NET_COL_SEP, MT2NET_LEAF_FIRST)
        pred = f" is {val_s}" if has_val else ""
        end = "." if MT2NET_TRAILING_PERIOD else ""
        if row_s and col_s:
            return f"For {row_s}, {col_s}{pred}{end}"
        label = col_s or row_s
        if label:
            return f"{label}{pred}{end}"
        return f"The value{pred}{end}"

    # STRUCTURAL — byte-identical to the retired length="long" preset
    row_s = join_path(row_path)
    col_s = join_path(col_path)
    clause = f"among {row_s}, " if row_s else ""
    what = f"the value of {col_s}" if col_s else "the value"
    pred = f" is {val_s}" if has_val else ""
    if title_s:
        return f"In the table '{title_s}', {clause}{what}{pred}."
    # str.capitalize() upper-cases the first character AND LOWER-CASES the rest,
    # so an untitled corpus had every header label case-folded -- 86,373 of
    # RealHiTBench's sentences, whose paths are 94.2% mixed-case. Only the
    # leading capital was ever intended. HiTab is unaffected either way: its
    # labels are already lower-case and 99.3% of its tables carry a title, so no
    # number measured on it moves.
    line = f"{clause}{what}{pred}."
    return line[:1].upper() + line[1:]
