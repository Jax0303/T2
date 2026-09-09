"""Name a retrieved cell by its rank; return that cell's stored value verbatim.

The reader never writes the answer string. It writes a rank, and the code looks
the rank up. Two independent ways to get a line's value are kept and checked
against each other by :func:`audit_value_map`:

  authoritative  re-render every corpus cell and key the sentence to
                 ``table.data[i][j]`` -- the value never passes through a parser.
  grammar        read the value back out of the frozen sentence with the
                 template's own grammar, for lines the corpus no longer renders
                 (frozen contexts predate a table-title fix; 128/31,620 on test).

The grammar reader is used only where the authoritative map has no entry, and
only after the two agree on every line where both speak.
"""
from __future__ import annotations
import re

#: ``In the table 'T', among ROW, the value of COL is VALUE.`` -- the value is
#: whatever sits after the last " is " and before the final period, so a number
#: inside the title or a header path can never be mistaken for it.
_TITLED = re.compile(r"^In the table '.* is (?P<v>.*)\.$")
#: Untitled compact form: ``ROW > COL: VALUE`` (or a bare value with no path).
_UNTITLED = re.compile(r"^(?:.*: )?(?P<v>[^:]*)$")


def value_from_sentence(line):
    """The value the template put at the end of this sentence, or None."""
    m = _TITLED.match(line)
    if m:
        return m.group("v")
    m = _UNTITLED.match(line)
    return m.group("v") if m else None


def build_value_map(data_dir, tids, page_titles, template="s3c"):
    """sentence -> stored cell value, straight from the grid (no parsing)."""
    from rag_agent.bench.hitab_grid import load_table
    from rag_agent.serialization.base import fmt_value
    from rag_agent.serialization.caption import with_page_title
    from rag_agent.serialization.templates import render, STRUCTURAL_COMPACT, MT2NET
    mode = STRUCTURAL_COMPACT if template == "s3c" else MT2NET
    out = {}
    for tid in sorted(tids):
        tab = load_table(tid, data_dir)
        if tab is None:
            continue
        t = tab.table
        title = with_page_title(tab.title, page_titles.get(tid))
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                v = t.data[i][j]
                if not str(v).strip():
                    continue
                s = render(mode, title, t.row_path(i), t.col_path(j), v)
                fv = fmt_value(v)
                # A sentence two cells share must agree on the value, or it is
                # not a key we may resolve through.
                if out.setdefault(s, fv) != fv:
                    out[s] = None
    return out


def audit_value_map(contexts, value_map):
    """Do the two readers agree wherever both speak? Run before any generation."""
    seen = agree = disagree = grammar_only = unreadable = 0
    examples = []
    for context in contexts:
        for line in context:
            seen += 1
            truth = value_map.get(line)
            guess = value_from_sentence(line)
            if truth is None:
                if guess is None:
                    unreadable += 1
                else:
                    grammar_only += 1
            elif truth == guess:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 5:
                    examples.append({"line": line, "authoritative": truth, "grammar": guess})
    return {"lines": seen, "agree": agree, "disagree": disagree,
            "grammar_only": grammar_only, "unreadable": unreadable,
            "disagreement_examples": examples}


def resolve(line, value_map):
    v = value_map.get(line)
    return v if v is not None else value_from_sentence(line)


def render_cellid(context):
    """The frozen lines, unchanged, each prefixed with its retrieval rank."""
    return "\n".join(f"[#{r}] {line}" for r, line in enumerate(context, 1))


_BRACKET = re.compile(r"\[\s*#?\s*(\d+)\s*\]")
_HASH = re.compile(r"#\s*(\d+)")
_INT_LIST = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")

#: Parsing rules, fixed before the run and chosen on dev. ``hash`` is the
#: 2026-09-10 v1 rule, kept so that run stays reproducible.
RULES = ("hash", "bracket", "bracket_bare")


def parse_ids(raw, rule):
    text = str(raw)
    if rule == "hash":
        found = _HASH.findall(text)
    elif rule in ("bracket", "bracket_bare"):
        found = _BRACKET.findall(text) + _HASH.findall(text)
        if not found and rule == "bracket_bare" and _INT_LIST.match(text):
            found = re.findall(r"\d+", text)
    else:
        raise ValueError(rule)
    return list(dict.fromkeys(int(x) for x in found))


def predict(raw, context, value_map, rule):
    """(prediction string, audit). Anything that is not a resolvable rank is dropped."""
    ids = parse_ids(raw, rule)
    audit = {"ids": ids, "n_ids": len(ids)}
    if not ids:
        return "", {**audit, "failure": "no_id"}
    values, bad = [], []
    for i in ids:
        if not 1 <= i <= len(context):
            bad.append(("out_of_range", i))
            continue
        v = resolve(context[i - 1], value_map)
        (values if v is not None else bad).append(v if v is not None else ("unresolved", i))
    if bad:
        audit["bad_ids"] = [i for _, i in bad]
        audit["failure"] = ("out_of_range" if any(k == "out_of_range" for k, _ in bad)
                            else "unresolved_value")
    if not values:
        audit["failure"] = audit.get("failure", "unresolved_value")
        return "", audit
    return ", ".join(values), audit
