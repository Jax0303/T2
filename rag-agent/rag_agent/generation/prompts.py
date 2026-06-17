"""Prompt builders for the two answer paths.

Both paths receive the *assembled context* from component 4 (operand cells, or
the whole table on fallback). The direct path asks the model to answer in
natural language; the codegen path asks it to compute the answer in Python over
a structured ``CELLS`` list, which isolates the model's arithmetic from its
reading — the comparison the thesis uses to show a weak reader still answers
when the right operands are supplied.
"""
from __future__ import annotations

from typing import List, Tuple

DIRECT_SYS = (
    "You answer questions about a table. You are given the relevant cells, each "
    "shown as its hierarchical header path and value. Use only the given cells. "
    "Reply with ONLY the final answer (a number or a short phrase), no units "
    "unless asked, no explanation."
)


def build_direct_user(query: str, context_text: str) -> str:
    return f"CONTEXT:\n{context_text}\n\nQUESTION: {query}\n\nANSWER:"


CODEGEN_SYS = (
    "You compute the answer to a table question with Python. You are given "
    "CELLS, a list of dicts each like {'path': [...], 'value': <str>}. Write a "
    "short Python snippet that computes the answer and assigns it to a variable "
    "named `answer`. Parse numbers from the string values yourself (strip commas "
    "/ %). Use ONLY the given CELLS. Output ONLY a Python code block, no prose."
)


def build_codegen_user(query: str, cells: List[dict]) -> str:
    lines = ["CELLS = ["]
    for c in cells:
        lines.append(f"    {{'path': {c['path']!r}, 'value': {c['value']!r}}},")
    lines.append("]")
    cells_repr = "\n".join(lines)
    return f"{cells_repr}\n\nQUESTION: {query}\n\n# Assign the result to `answer`."


def cells_from_chunks(chunks) -> List[dict]:
    """Turn retrieved S2 cell chunks into {'path', 'value'} dicts for codegen.

    The value is the text after the last ': ' on the cell's last line (the S2
    ``path: value`` rendering); the path is the chunk's recorded header path.
    """
    out: List[dict] = []
    for ch in chunks:
        last = ch.text.splitlines()[-1] if ch.text else ""
        value = last.rsplit(": ", 1)[-1] if ": " in last else last
        path = ch.header_paths[0] if ch.header_paths else []
        out.append({"path": list(path), "value": value})
    return out
