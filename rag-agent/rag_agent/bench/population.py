# SPDX-License-Identifier: MIT
"""Frozen query populations: the set of questions an experiment runs on, on disk.

Every script used to derive its own population at run time -- load the split,
drop what the reconstructor cannot build a header tree for, shuffle, take the
first --n. That makes the population a FUNCTION OF THE CODE: the two header
fixes (``eceea04``, ``b10818d``) changed which tables reconstruct, so the "same"
n=100 silently became a different 100 queries and old numbers stopped being
paired with new ones.

Freezing splits those two things apart. The population is derived once, written
here, and committed; afterwards a code change can move the METRIC but never the
membership. :func:`pin` is the whole enforcement: if a frozen query has gone
missing from what the current code derives, that is a loud error rather than a
quietly smaller n.

Files live in ``populations/`` (not ``data/``, which is gitignored -- a frozen
population that is not committed does not freeze anything).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

DIR = Path(__file__).resolve().parents[2] / "populations"


def path(name: str) -> Path:
    return DIR / f"{name}.txt"


def write(name: str, query_ids: Sequence[str], meta: dict) -> Path:
    """One query_id per line, derivation conditions in a ``#`` header."""
    DIR.mkdir(parents=True, exist_ok=True)
    p = path(name)
    lines = [f"# {name}", f"# {json.dumps(meta, ensure_ascii=False, sort_keys=True)}",
             f"# n={len(query_ids)}"]
    lines += list(query_ids)
    p.write_text("\n".join(lines) + "\n")
    return p


def read(name: str) -> tuple[list[str], dict] | None:
    """(query_ids, meta), or None when this population has not been frozen."""
    p = path(name)
    if not p.exists():
        return None
    ids, meta = [], {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body.startswith("{"):
                meta = json.loads(body)
            continue
        ids.append(line)
    return ids, meta


def pin(name: str, queries: Iterable, strict: bool = True) -> list:
    """Reorder ``queries`` to the frozen list for ``name``; unfrozen -> unchanged.

    ``strict`` (the default) refuses to run on a population that has drifted:
    a frozen query the current code no longer derives is exactly the failure
    this module exists to catch, and silently continuing with a smaller n is
    how the old numbers became unpairable.
    """
    frozen = read(name)
    queries = list(queries)
    if frozen is None:
        return queries
    ids, _ = frozen
    # AIT-QA / RealHiTBench build their queries as dicts, HiTab as objects
    have = {(q["query_id"] if isinstance(q, dict) else q.query_id): q
            for q in queries}
    missing = [i for i in ids if i not in have]
    if missing and strict:
        raise RuntimeError(
            f"population {name!r}: {len(missing)}/{len(ids)} frozen queries are not "
            f"derivable by the current code (first: {missing[:3]}). Either the "
            f"derivation regressed, or the population must be re-frozen on purpose "
            f"with scripts/freeze_populations.py --force.")
    return [have[i] for i in ids if i in have]
