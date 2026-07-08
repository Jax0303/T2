"""Extract and safely execute model-generated Python for the codegen path.

The model returns a snippet that assigns ``answer`` over an injected ``CELLS``
list. We run it in a separate Python process (``sys.executable``) with a wall
clock timeout and a stripped environment, then read back ``repr(answer)`` from
stdout. Execution is best-effort sandboxed — enough for benchmark runs, not a
hostile-code sandbox.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the first fenced code block, or return the raw text if unfenced."""
    if not text:
        return ""
    m = _CODE_FENCE.search(text)
    return (m.group(1) if m else text).strip()


@dataclass
class CodeResult:
    ok: bool
    value: Optional[str]
    error: str = ""


# Runner: defines CELLS, execs the model snippet, prints repr(answer).
_RUNNER = """\
import json, sys
CELLS = json.loads(sys.argv[1])
_ns = {"CELLS": CELLS}
try:
    exec(compile(sys.argv[2], "<codegen>", "exec"), _ns)
    print(repr(_ns.get("answer")))
except Exception as e:
    sys.stderr.write(type(e).__name__ + ": " + str(e))
    sys.exit(1)
"""


def run_codegen(code: str, cells: List[dict], timeout: float = 5.0) -> CodeResult:
    """Execute ``code`` against ``cells`` in a subprocess, return ``answer``."""
    code = extract_code(code) if "```" in code else code.strip()
    if not code:
        return CodeResult(ok=False, value=None, error="empty code")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, json.dumps(cells), code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        return CodeResult(ok=False, value=None, error="timeout")
    if proc.returncode != 0:
        return CodeResult(ok=False, value=None, error=(proc.stderr or "").strip()[:200])
    out = proc.stdout.strip()
    # repr(None) -> "None"; treat as no answer
    if out == "None" or out == "":
        return CodeResult(ok=False, value=None, error="answer is None")
    # strip surrounding quotes from repr of a string
    if len(out) >= 2 and out[0] in "'\"" and out[-1] == out[0]:
        out = out[1:-1]
    return CodeResult(ok=True, value=out)
