# SPDX-License-Identifier: MIT
"""S2_shuf must be the same permutation in every process.

RESEARCH_STRUCTURE.md §6: it used ``hash(tuple(segs))``, and str hashing is
salted per process, so claim 4 (S2 == S2_shuf to 4 decimals) was not reproducible.
The salt is exactly what this test varies.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = (
    "import sys; sys.path.insert(0, 'scripts');"
    "from operand_collision_multihiertt import cell_text;"
    "c = {'row_path': ['a', 'bb', 'ccc'], 'col_path': ['dddd', 'e'], 'value': '7'};"
    "print(cell_text(c, 'S2_shuf'))"
)


def _run(hashseed: str) -> str:
    return subprocess.run([sys.executable, "-c", SNIPPET], cwd=ROOT, check=True,
                          capture_output=True, text=True,
                          env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
                          ).stdout.strip()


def test_shuffle_survives_hash_salt():
    out = {_run(seed) for seed in ("0", "1", "12345")}
    assert len(out) == 1, f"S2_shuf differs across PYTHONHASHSEED: {out}"
    text = out.pop()
    assert sorted(text.split(": ")[0].split(" > ")) == ["a", "bb", "ccc", "dddd", "e"]
    assert text.endswith(": 7")
