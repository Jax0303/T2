# SPDX-License-Identifier: MIT
"""A stored ``pred`` must be the text that was scored.

``rescore_currency.py`` recomputes ``answer_em`` from the ``pred`` field of
``results/*_records.jsonl``, so a prediction stored truncated makes the file
unscorable from its own contents. 581 of 104,382 stored predictions had been
cut at 120 characters -- concentrated in the ``dump``/``cascade`` baselines,
which answer verbosely -- so a rescore silently penalised exactly the arms the
paper compares itself against. This fails if the slice comes back.
"""
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
# "pred": raw[:120]   /   rec["pred"] = out_txt[:120]   /   ..., out_txt[:120]
_TRUNCATED_PRED = re.compile(r'"pred"\s*[:=]\s*[A-Za-z_][\w.\[\]"\']*\[:\s*\d+\s*\]')


def test_no_script_stores_a_truncated_prediction():
    offenders = []
    for py in sorted(SCRIPTS.glob("*.py")):
        for n, line in enumerate(py.read_text(errors="replace").splitlines(), 1):
            if _TRUNCATED_PRED.search(line):
                offenders.append(f"{py.name}:{n}: {line.strip()}")
    assert not offenders, "predictions stored truncated:\n" + "\n".join(offenders)
