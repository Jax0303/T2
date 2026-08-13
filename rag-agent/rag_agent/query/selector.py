# SPDX-License-Identifier: MIT
"""Strip the selector clause off a two-part table question.

A large slice of MultiHiertt's arithmetic questions name two different things:
a **selector** that picks which year/section to look in, and the **target**
whose cells are actually the gold operands.

    In the year with largest amount of Net revenues, what is the growth rate
    of Pretax income?
    └──────────────── selector ────────────────┘  └──── target ────┘

A bi-encoder reads the whole string as one vector and happily retrieves the
selector's cells, which are never gold. Measured on the within-document ladder,
the single largest failure bucket (56% of rank-1 misses) is exactly this: the
top hit sits in the right table on the wrong row.

The clause is regular enough to cut lexically -- no LLM, no model. The selector
is not discarded by callers that need it (which year *was* the largest is a
separate lookup); this module only produces the query used for **operand**
retrieval.
"""
from __future__ import annotations

import re

# "year", "section", "year/section", "reporting period" ...
_AXIS = r"(?:year|section|period|quarter|month|column|row|table)s?(?:\s*/\s*\w+)?"
# what makes it a SELECTOR rather than a plain period constraint ("in the year
# 2009"): a relative clause comparing magnitudes.
_REL = r"(?:with|where|that|which|showed|for|having)"
_WH = r"(?:what|how|which|by how much|calculate)"

# leading: "In the <axis> <rel> ..., <target>"  /  "... <axis> <rel> ... <wh> ..."
_HEAD = re.compile(
    rf"^\s*in\s+the\s+(?:[\w&/-]+\s+)*?{_AXIS}\s+{_REL}\b"
    rf"(?:[^,]*?,\s*|.*?(?=\b{_WH}\b))", re.I)
# trailing: "<target> in the <axis> <rel> ...?"
_TAIL = re.compile(
    rf",?\s*\bin\s+the\s+(?:[\w&/-]+\s+)*?{_AXIS}\s+{_REL}\b.*$", re.I)

_TIDY = re.compile(r"\s{2,}")


def has_selector(question: str) -> bool:
    """Does this question pick its scope by comparing some *other* quantity?"""
    return bool(_HEAD.search(question) or _TAIL.search(question))


def strip_selector(question: str) -> str:
    """Return the target half of the question, or the question unchanged.

    Never returns an empty or near-empty string: if cutting leaves too little
    to retrieve on, the original is kept, because a two-word query is worse
    than a diluted one.
    """
    out = _HEAD.sub("", question, count=1)
    out = _TAIL.sub("", out, count=1)
    out = _TIDY.sub(" ", out).strip(" ,")
    if not out.endswith("?") and question.rstrip().endswith("?"):
        out += "?"
    # ponytail: 4-token floor, tuned by eye on the 72 matching MultiHiertt
    # queries. Raise it if a corpus with terser targets starts losing recall.
    return question if len(out.split()) < 4 else out


def _demo() -> None:
    cases = [
        # leading selector, comma-delimited
        ("In the year with largest amount of Net revenues, what is the growth "
         "rate of Pretax income? (in %)",
         "what is the growth rate of Pretax income? (in %)"),
        # leading selector, NO comma before the wh-word
        ("In the year with largest amount of North America what's the "
         "increasing rate of Middle East & Asia ?",
         "what's the increasing rate of Middle East & Asia ?"),
        # trailing selector
        ("What is the growing rate of Mortgage loans in the year with the most "
         "Equity securities?",
         "What is the growing rate of Mortgage loans?"),
        # axis written as a pair
        ("In the year/section with the most Undrawn commitments to extend "
         "credit, what is the growth rate of Financial standby letters of credit?",
         "what is the growth rate of Financial standby letters of credit?"),
    ]
    for q, want in cases:
        got = strip_selector(q)
        assert got == want, f"\n  in   {q!r}\n  got  {got!r}\n  want {want!r}"
        assert has_selector(q), q

    # a plain period constraint is NOT a selector -- cutting it would throw away
    # the only thing that separates the 2011 column from the 2012 one
    for q in ("what is the growth rate of net revenue in 2012?",
              "In 2009, what was the total of Operating revenues?",
              "by how much did the low of mktx stock increase from 2011 to march 2012?"):
        assert not has_selector(q), q
        assert strip_selector(q) == q, q

    # cutting must never leave a stub
    stub = "In the year with the most Revenues, what?"
    assert strip_selector(stub) == stub
    print("selector: ok")


if __name__ == "__main__":
    _demo()
