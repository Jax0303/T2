# SPDX-License-Identifier: MIT
"""Answer generation over retrieved context, with two paths.

  * ``direct``  — the LLM reads the retrieved chunks and answers in one line.
  * ``codegen`` — the LLM emits a short Python snippet that assigns ``answer``
    from numbers it reads in the context; the snippet is executed in a guarded
    namespace (AST-whitelisted, no imports/attributes/calls except a small math
    allow-list) and falls back to ``direct`` parsing if it fails.

Token budget is enforced on the assembled context (``max_context_tokens``,
default 4096 ≈ a 7B model's working window) so a full-table fallback chunk cannot
overflow the prompt. Evaluation reuses the repo's ``numeric_match`` / ``exact_match``.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..bench.schema import Chunk
from ..eval.metrics import numeric_match, exact_match

_ALLOWED_FUNCS = {
    "sum": sum, "abs": abs, "round": round, "min": min, "max": max, "len": len,
    "float": float, "int": int,
}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# rough token estimate: ~4 chars/token
_CHARS_PER_TOKEN = 4


def format_context(
    chunks: Sequence[Chunk], max_context_tokens: int = 4096, return_meta: bool = False
):
    """Join chunk texts under a token budget (truncating low-priority tail).

    ``return_meta=True`` also returns whether any chunk was dropped — a caller
    appending high-value content at the *end* of ``chunks`` (e.g. injected
    total rows) needs to know when that content silently missed the budget
    instead of assuming every chunk it built made it into the prompt.
    """
    budget = max_context_tokens * _CHARS_PER_TOKEN
    out, used = [], 0
    truncated = False
    for ch in chunks:
        line = ch.text
        if used + len(line) > budget and out:
            truncated = True
            break
        out.append(line)
        used += len(line) + 1
    text = "\n".join(out)
    return (text, truncated) if return_meta else text


# ---------------------------------------------------------------------------
# Guarded code execution (codegen path)
# ---------------------------------------------------------------------------

class _Guard(ast.NodeVisitor):
    """Reject imports, attribute access, calls outside the allow-list, and loops.

    The codegen prompt asks for a single ``answer = <arithmetic expression>``
    line — loops are never a valid response, only a way for a hallucinated
    snippet to hang the (in-process, untimed) exec call.
    """

    def visit_Import(self, node):           # noqa: N802
        raise ValueError("import not allowed")

    visit_ImportFrom = visit_Import

    def visit_Attribute(self, node):        # noqa: N802
        raise ValueError("attribute access not allowed")

    def visit_While(self, node):            # noqa: N802
        raise ValueError("loops not allowed")

    def visit_For(self, node):              # noqa: N802
        # only the `for` *statement* is an ast.For node; comprehension
        # generators (ast.comprehension) are untouched and still work.
        raise ValueError("loops not allowed")

    def visit_Call(self, node):             # noqa: N802
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError(f"call not allowed: {ast.dump(node.func)}")
        self.generic_visit(node)


def _safe_exec(code: str) -> Optional[float]:
    """Execute ``code`` (which should set ``answer``) in a guarded namespace."""
    tree = ast.parse(code, mode="exec")
    _Guard().visit(tree)
    ns = dict(_ALLOWED_FUNCS)
    exec(compile(tree, "<codegen>", "exec"), {"__builtins__": {}}, ns)  # noqa: S102
    val = ns.get("answer")
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    block = m.group(1) if m else text
    lines = []
    for ln in block.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("print(") or s.startswith("print "):
            continue  # drop blanks, comments, and print() noise (a frequent reject cause)
        lines.append(ln)
    return "\n".join(lines)


def _parse_number(text: str) -> Optional[float]:
    m = _NUM_RE.findall(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    answer: object
    raw: str
    mode: str
    used_codegen: bool = False
    context_truncated: bool = False


_DIRECT_SYS = (
    "You answer questions about a table. Use ONLY the rows given. "
    "Reply with the final answer only — a number or a short phrase, no explanation."
)
_CODEGEN_SYS = (
    "You answer table questions by writing Python. Do NOT rebuild the table or "
    "create lists/dicts. Read only the few numbers you need directly from the "
    "rows and write ONE line: `answer = <arithmetic expression over those "
    "numbers>`. Use only +,-,*,/, parentheses and sum/abs/round/min/max/len. "
    "No print(), no comments. Return a single ```python``` block with just that line."
)


def answer(
    question: str,
    chunks: Sequence[Chunk],
    llm,
    mode: str = "direct",
    max_context_tokens: int = 4096,
    max_tokens: int = 256,
    codegen_max_tokens: int = 160,
) -> AnswerResult:
    """Generate an answer for ``question`` from the retrieved ``chunks``.

    ``codegen_max_tokens`` caps the codegen completion; reasoning models
    (gpt-oss, qwen3) spend completion tokens on hidden reasoning before the
    code block, so they need a much higher cap than the 160 default.
    """
    ctx, truncated = format_context(chunks, max_context_tokens, return_meta=True)
    if mode == "codegen":
        user = f"ROWS:\n{ctx}\n\nQUESTION: {question}\n\nOne line: answer = ..."
        raw = llm.complete(system=_CODEGEN_SYS, user=user,
                           max_tokens=codegen_max_tokens)
        try:
            val = _safe_exec(_extract_code(raw))
            if val is not None:
                return AnswerResult(answer=val, raw=raw, mode=mode, used_codegen=True,
                                    context_truncated=truncated)
        except Exception:
            # Any generated-code failure (bad index, overflow, recursion, ...)
            # falls back to direct-text parsing below — it must never escape
            # answer() uncaught, or callers that treat exceptions as an
            # API-quota cutoff (e.g. answer_accuracy_injection.py) will
            # misattribute a codegen bug as a rate limit and drop the rest of
            # the run.
            pass
        return AnswerResult(answer=_parse_number(raw), raw=raw, mode=mode, used_codegen=False,
                            context_truncated=truncated)

    user = f"ROWS:\n{ctx}\n\nQUESTION: {question}\n\nAnswer:"
    raw = llm.complete(system=_DIRECT_SYS, user=user, max_tokens=max_tokens)
    num = _parse_number(raw)
    return AnswerResult(answer=num if num is not None else raw.strip(), raw=raw, mode=mode,
                        context_truncated=truncated)


def evaluate_answer(pred, gold, rel_tol: float = 0.02) -> bool:
    """True if ``pred`` matches ``gold`` numerically (±tol) or by exact string."""
    if numeric_match(pred, gold, rel_tol=rel_tol):
        return True
    return exact_match(pred, gold)
