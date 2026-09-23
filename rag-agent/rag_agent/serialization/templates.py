# SPDX-License-Identifier: MIT
"""The cell->sentence templates, and nothing else.

Replaces the ``short``/``medium``/``long`` preset axis. Sentence length is no
longer an experimental variable; what remains is the one contrast that carries a
claim:

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
  nothing while still spending budget. This template was pre-registered in
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
"""
from __future__ import annotations

from typing import Sequence

from .base import fmt_value, join_path

STRUCTURAL = "structural"
STRUCTURAL_COMPACT = "structural_compact"
STRUCTURAL_LEAF = "structural_leaf"
STRUCTURAL_LEAF_X2 = "structural_leaf_x2"
TEMPLATES = (STRUCTURAL, STRUCTURAL_COMPACT, STRUCTURAL_LEAF, STRUCTURAL_LEAF_X2)


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
        # nothing and still costs tokens, and the budget pays in cells that fit.
        # Fall back to the S2 path string, byte-identical to
        # point3_reconstruction_cost.cell_text(...,"S2").
        path = join_path([*row_path, *col_path])
        if not has_val:
            return path
        return f"{path}: {val_s}" if path else val_s

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
