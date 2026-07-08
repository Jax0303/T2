# SPDX-License-Identifier: MIT
"""Header-Path Intent Resolution (HPIR).

Motivation (measured, see ``docs/FAILURE_ANALYSIS.md`` / ``docs/VERDICT.md``):
the two bottlenecks of the split-store table-RAG pipeline share one root cause —
**the hierarchical header path the query refers to is never made explicit**.

  * Retrieval bottleneck: the best retriever is the one over *header-path*
    serialisations (``dense_header_path`` R@1 0.49 ≫ ``plain`` 0.39), yet it is
    fed the raw narrative question, most tokens of which are not header text.
  * Answer bottleneck: 82% of answer failures are "code runs fine but binds to
    the wrong cell" — silent grounding errors from free-guessed header strings.

HPIR is a single query-understanding step that maps the question to a structured
header-path intent and applies it in two regimes:

  1. ``expand_for_retrieval``  — corpus-free: strip narrative, keep header-likely
     tokens (+ operation hints) → a pseudo-document aligned with the header-path
     index. Boosts retrieval without touching the answer stage.
  2. ``resolve_against_table`` — table-grounded: rank the retrieved table's
     *actual* row/column header paths against the query, returning concrete
     (row_path, col_path) bindings. The bindings are validated to *exist* in the
     table, which is exactly the signal the free-guess extractor lacks.

The module is deterministic by default (no LLM, fully unit-testable). An optional
LLM can refine each regime, but its output is always validated against the
table's real header inventory and falls back to the deterministic result, so the
LLM can never introduce a non-existent binding.

Related work the framing builds on (real): HyDE (Gao et al., 2023) and query2doc
(Wang et al., 2023) for query expansion; DTR (Herzig et al., NAACL 2021) and
header-path serialisation for table retrieval; PAL (Gao et al., 2023) and
Self-Debugging (Chen et al., ICLR 2024) for program-grounded answering. HPIR's
novelty is using the *hierarchical header path* as the explicit shared IR across
both stages of a split-store retriever.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..router.query_classifier import QueryType, classify_query
from ..stores.original_store import OriginalTable

# ---------------------------------------------------------------------------
# Tokenisation / term extraction
# ---------------------------------------------------------------------------

# Words that signal the *operation* rather than a header target. Dropping them
# from the retrieval expansion keeps the pseudo-document close to header text.
_OP_CUE_WORDS = {
    "increase", "increased", "decrease", "decreased", "rise", "rose", "drop",
    "dropped", "grew", "grow", "change", "changed", "difference", "differ",
    "gap", "ratio", "fraction", "proportion", "percentage", "percent", "share",
    "sum", "total", "combined", "average", "mean", "range", "spread", "times",
    "more", "less", "fewer", "higher", "lower", "greater", "smaller", "larger",
    "highest", "lowest", "largest", "smallest", "maximum", "minimum", "most",
    "least", "biggest", "best", "worst", "peak", "top", "bottom", "compared",
    "than", "versus", "vs", "exceed", "exceeded", "above", "below", "count",
    "number", "how", "many", "much", "what", "which", "who", "where", "when",
    "value", "amount", "rate",
}

# Generic stopwords (subset shared with OriginalDB; kept local to avoid coupling
# to a script-level constant).
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "from", "by", "with",
    "and", "or", "is", "was", "were", "be", "been", "are", "as", "it", "this",
    "that", "do", "does", "did", "has", "have", "had", "then", "into", "their",
    "there", "these", "those", "they", "them", "its", "his", "her", "our", "we",
    "but", "not", "all", "any", "each", "per", "out", "over", "about", "also",
    "between", "among", "within", "across", "during", "while", "whose", "whom",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_&$%/-]+|\d{4}")  # words or 4-digit years
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")           # plausible year tokens


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def extract_target_terms(query: str) -> List[str]:
    """Content tokens that are *candidate header text* (order-preserving, unique).

    Drops stopwords and operation cue-words but **keeps year tokens** — in HiTab,
    years are frequently column headers (``current $millions > 2014``), so they
    are signal for header-path retrieval (unlike free numeric-cell matching,
    which ``VERDICT.md`` showed hurts).
    """
    seen: set = set()
    out: List[str] = []
    for tok in _tokenize(query):
        is_year = bool(_YEAR_RE.fullmatch(tok))
        if not is_year:
            if tok in _STOPWORDS or tok in _OP_CUE_WORDS:
                continue
            if len(tok) < 3:
                continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Structured intent
# ---------------------------------------------------------------------------

@dataclass
class HeaderPathIntent:
    """Structured query intent — the shared header-path IR."""

    operation: str                      # QueryType value (e.g. "arithmetic_agg")
    needs_symbolic: bool                # arithmetic → symbolic/codegen path
    target_terms: List[str]             # header-candidate tokens from the query
    expansion: str                      # retrieval pseudo-document
    col_paths: List[List[str]] = field(default_factory=list)  # table-grounded
    row_paths: List[List[str]] = field(default_factory=list)  # table-grounded
    source: str = "deterministic"       # "deterministic" | "llm" | "llm+fallback"

    def binding_hint(self, max_each: int = 4) -> str:
        """Human/LLM-readable suggested bindings for the grounded prompt."""
        rows = [" > ".join(p) for p in self.row_paths[:max_each] if p]
        cols = [" > ".join(p) for p in self.col_paths[:max_each] if p]
        lines = [f"OPERATION: {self.operation}"]
        if rows:
            lines.append("LIKELY ROW HEADERS: " + " | ".join(rows))
        if cols:
            lines.append("LIKELY COL HEADERS: " + " | ".join(cols))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime 1 — corpus-free retrieval expansion
# ---------------------------------------------------------------------------

# Operation → header-likely hint tokens that the gold table's header path tends
# to contain (e.g. a ratio question's table usually has a "total" header).
_OP_HINTS = {
    QueryType.ARITHMETIC_AGG: ["total", "percent"],
    QueryType.MULTI_OP_FORMULA: ["total", "percent"],
    QueryType.COMPARISON_OR_COUNT: ["total"],
}


def expand_for_retrieval(query: str, intent: Optional[HeaderPathIntent] = None) -> str:
    """Build a header-path-aligned pseudo-document for retrieval.

    Returns a space-joined token string usable by either a dense embedder or a
    keyword retriever. Deterministic; no table needed.
    """
    qintent = classify_query(query)
    terms = extract_target_terms(query)
    hints = _OP_HINTS.get(qintent.qtype, [])
    # Keep target terms first (highest weight under bag-of-words), then op hints.
    tokens = terms + [h for h in hints if h not in terms]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Regime 2 — table-grounded resolution
# ---------------------------------------------------------------------------

def _rank_paths(
    table: OriginalTable, terms: Sequence[str], axis: str, top_n: int
) -> List[List[str]]:
    """Rank an axis's distinct header paths by fuzzy overlap with query terms."""
    n = table.n_cols if axis == "col" else table.n_rows
    path_of = table.col_path if axis == "col" else table.row_path
    query_str = " ".join(terms)
    scored: List[Tuple[float, str, List[str]]] = []
    seen: set = set()
    for i in range(n):
        p = path_of(i)
        if not p:
            continue
        key = " > ".join(p)
        if key in seen:
            continue
        seen.add(key)
        s = table._fuzzy_score(query_str, p)  # reuse store's token+similarity scorer
        if s > 0:
            scored.append((s, key, p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, _, p in scored[:top_n]]


def resolve_against_table(
    query: str,
    table: OriginalTable,
    top_n_cols: int = 3,
    top_n_rows: int = 4,
) -> HeaderPathIntent:
    """Resolve the query to concrete header paths that *exist* in ``table``.

    Deterministic: ranks the table's real top/left header paths against the
    query's target terms using the store's own fuzzy scorer, so every returned
    binding is guaranteed to be a real header path (no free guessing).
    """
    qintent = classify_query(query)
    terms = extract_target_terms(query)
    col_paths = _rank_paths(table, terms, "col", top_n_cols)
    row_paths = _rank_paths(table, terms, "row", top_n_rows)
    return HeaderPathIntent(
        operation=qintent.qtype.value,
        needs_symbolic=qintent.needs_symbolic,
        target_terms=terms,
        expansion=expand_for_retrieval(query),
        col_paths=col_paths,
        row_paths=row_paths,
        source="deterministic",
    )


# ---------------------------------------------------------------------------
# Optional LLM refinement (validated against the real header inventory)
# ---------------------------------------------------------------------------

_LLM_RESOLVE_SYS = (
    "You map a table question to the header paths it refers to. "
    "You are given the table's ROW HEADERS and COL HEADERS. "
    "Return JSON {\"row_headers\": [...], \"col_headers\": [...]} choosing ONLY "
    "from the given lists (copy strings exactly). No commentary, JSON only."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _distinct_paths(table: OriginalTable, axis: str, cap: int = 60) -> List[List[str]]:
    n = table.n_cols if axis == "col" else table.n_rows
    path_of = table.col_path if axis == "col" else table.row_path
    seen, out = set(), []
    for i in range(n):
        p = path_of(i)
        key = " > ".join(p)
        if p and key not in seen:
            seen.add(key)
            out.append(p)
        if len(out) >= cap:
            break
    return out


def resolve_intent(
    query: str,
    table: OriginalTable,
    llm=None,
    top_n_cols: int = 3,
    top_n_rows: int = 4,
) -> HeaderPathIntent:
    """Table-grounded resolution, optionally refined by an LLM.

    Always starts from the deterministic resolution. If ``llm`` is given, asks it
    to pick header paths from the real inventory and *intersects* the answer with
    paths that actually exist; any hallucinated path is dropped and the
    deterministic ranking backfills, so bindings remain grounded.
    """
    base = resolve_against_table(query, table, top_n_cols, top_n_rows)
    if llm is None:
        return base

    import json

    rows_inv = [" > ".join(p) for p in _distinct_paths(table, "row")]
    cols_inv = [" > ".join(p) for p in _distinct_paths(table, "col")]
    user = (
        f"Question: {query}\n\n"
        f"ROW HEADERS:\n- " + "\n- ".join(rows_inv) + "\n\n"
        f"COL HEADERS:\n- " + "\n- ".join(cols_inv) + "\n\nJSON only."
    )
    try:
        raw = llm.complete(system=_LLM_RESOLVE_SYS, user=user, max_tokens=200)
    except Exception:
        return base
    m = _JSON_RE.search(raw or "")
    if not m:
        return base
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return base

    def _valid(cands, axis) -> List[List[str]]:
        out, seen = [], set()
        for s in cands or []:
            for real in _distinct_paths(table, axis):
                key = " > ".join(real)
                if key in seen:
                    continue
                # accept if the LLM string matches (substring either way) a real path
                if table._match_path(str(s), real) or str(s).strip().lower() in key.lower():
                    out.append(real)
                    seen.add(key)
                    break
        return out

    llm_rows = _valid(obj.get("row_headers"), "row")
    llm_cols = _valid(obj.get("col_headers"), "col")

    # Backfill from deterministic ranking so we never regress to empty.
    def _merge(primary, fallback, cap):
        out, seen = [], set()
        for p in primary + fallback:
            key = " > ".join(p)
            if key not in seen:
                seen.add(key)
                out.append(p)
            if len(out) >= cap:
                break
        return out

    base.row_paths = _merge(llm_rows, base.row_paths, top_n_rows)
    base.col_paths = _merge(llm_cols, base.col_paths, top_n_cols)
    base.source = "llm" if (llm_rows or llm_cols) else "llm+fallback"
    return base
