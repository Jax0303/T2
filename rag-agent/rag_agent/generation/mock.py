"""A deterministic LLM stub for smoke tests and CI (no API key, no model).

It is not a reasoner — it returns the most plausible single-cell answer so the
generation/eval plumbing can be exercised end to end:

* direct  → the value of the first cell in the context.
* codegen → a snippet that returns the first cell's numeric value.
"""
from __future__ import annotations

import re

from ..llm.base import BaseLLM

# Value after the first "path: value" line. ``[ \t]*`` (not ``\s*``) so the
# newline is never consumed and we don't skip onto the following line.
_CTX_VALUE = re.compile(r":[ \t]+([^\n]+)")


class MockLLM(BaseLLM):
    name = "mock"

    def complete(self, system: str, user: str, max_tokens: int = 256) -> str:
        if "CELLS" in user and "answer" in user:
            # codegen prompt: emit code returning the first cell's number
            return (
                "```python\n"
                "def _num(s):\n"
                "    import re\n"
                "    m = re.search(r'-?\\d[\\d,]*\\.?\\d*', str(s))\n"
                "    return float(m.group(0).replace(',', '')) if m else None\n"
                "answer = _num(CELLS[0]['value']) if CELLS else None\n"
                "```"
            )
        # direct prompt: echo the first context value
        m = _CTX_VALUE.search(user)
        return m.group(1).strip() if m else ""
