# SPDX-License-Identifier: MIT
"""The multi-cell pool must count ANSWER cells, not quantity_link operands.

gold_operands pools every bucket of quantity_link, so it carries the numbers the
source sentence cites as well as the answer. Defining the pool on it put 47
single-answer queries into an 81-query "multi-cell" population.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

POPS = {s: ROOT / f"populations/hitab_{s}_lookup_multi.txt"
        for s in ("dev", "test", "train")}


def _ids(p):
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]


def _raw(split):
    f = ROOT / f"data/hitab/data/{split}_samples.jsonl"
    return {str(json.loads(l)["id"]): json.loads(l) for l in open(f)}


def _n_answer(d):
    return len(((d["linked_cells"].get("quantity_link") or {}).get("[ANSWER]") or {}))


def test_every_member_answers_with_at_least_two_cells():
    for split, p in POPS.items():
        raw = _raw(split)
        for qid in _ids(p):
            assert _n_answer(raw[qid]) >= 2, f"{split}/{qid} answers with one cell"


def test_no_member_computes_its_answer():
    """aggregation != none is the arithmetic pool, not this one."""
    for split, p in POPS.items():
        raw = _raw(split)
        for qid in _ids(p):
            assert raw[qid]["aggregation"] == ["none"], f"{split}/{qid} aggregates"


def test_disjoint_from_the_single_cell_pool():
    single = set(_ids(ROOT / "populations/hitab_dev_lookup_all.txt"))
    assert not (set(_ids(POPS["dev"])) & single)


def test_header_records_the_answer_cell_histogram():
    """The pool's whole point is the >=2 spread; a header without it hides it."""
    for p in POPS.values():
        meta = json.loads(next(l for l in open(p) if l.startswith("# {"))[2:])
        hist = meta["answer_cells_hist"]
        assert hist and min(int(k) for k in hist) >= 2
        assert sum(hist.values()) == meta["n"] == len(_ids(p))
