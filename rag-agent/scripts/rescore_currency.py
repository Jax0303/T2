#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-score every committed run through the currency fix in `_hmt_str_to_float`.

``answer_em`` is a DERIVED field: the records carry the model's ``pred``, and the
scorer turns it into a 0/1. When the scorer is corrected the stored 0/1 stops
matching the prediction it came from, so the file is no longer reproducible from
its own contents. This recomputes it from ``pred`` -- nothing is invented, the
same predictions are read with the fixed rule.

Rewrites, for every affected run:
  * ``*_records.jsonl``      -- ``answer_em`` per arm
  * the run's summary JSON   -- ``summary[arm].answer_em`` and every
                                ``paired_tests["answer_em:..."]`` entry
and writes a ledger of every change to ``results/currency_rescore.json``.

Cross-run paired files (``scripts/paired_em_between_runs.py`` output) are NOT
touched here: regenerate them from the rescored records with that script.

    PYTHONPATH=.:scripts .venv/bin/python scripts/rescore_currency.py --dry-run
    PYTHONPATH=.:scripts .venv/bin/python scripts/rescore_currency.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent.eval.metrics import hitab_exact_match_text  # noqa: E402
from manual_sentence_ceiling import mcnemar  # noqa: E402


def gold_maps(data_dir: str) -> dict:
    """corpus -> {query_id: gold answer}. Built once; the corpora are the slow part."""
    from corpus_dump_vs_cell import (aitqa_corpus, hitab_corpus,
                                     multihiertt_corpus, realhitbench_corpus)
    out = {}
    for name, build in (
            ("hitab_dev", lambda: hitab_corpus(data_dir, "dev", "hitab_dev_lookup_all")),
            ("hitab_test", lambda: hitab_corpus(data_dir, "test", "hitab_test_lookup_all")),
            ("aitqa", lambda: aitqa_corpus(pin=False)),
            ("realhitbench", lambda: realhitbench_corpus(pin=False)),
            ("multihiertt", lambda: multihiertt_corpus(400, 42))):
        try:
            out[name] = {q["query_id"]: q["answer"] for q in build().queries}
        except Exception as e:                       # a split that is not present
            print(f"  [skip] {name}: {e}")
    return out


def pick_gold(recs: list, maps: dict):
    """Whichever corpus covers most of this file's query ids, if it covers half."""
    best, cov = None, 0
    for name, g in maps.items():
        c = sum(1 for r in recs if r.get("query_id") in g)
        if c > cov:
            best, cov = name, c
    return (best, maps[best]) if best and cov >= len(recs) * 0.5 else (None, None)


def arms_of(recs: list) -> list:
    return sorted({k for r in recs for k, v in r.items()
                   if isinstance(v, dict) and "pred" in v})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--results", default="results")
    ap.add_argument("--apply", action="store_true", help="write the files")
    ap.add_argument("--out", default="results/currency_rescore.json")
    args = ap.parse_args()

    print("building gold maps ...")
    maps = gold_maps(args.data_dir)
    ledger = {"fix": "strip '$' in _hmt_str_to_float, both sides",
              "applied": bool(args.apply), "runs": {}}

    for rf in sorted(Path(args.results).glob("*_records.jsonl")):
        try:
            recs = [json.loads(l) for l in rf.read_text().splitlines() if l.strip()]
        except Exception:
            continue
        if not recs:
            continue
        corpus, gold = pick_gold(recs, maps)
        if gold is None:
            continue
        changed, per_arm = 0, {}
        for arm in arms_of(recs):
            before = after = n = 0
            for r in recs:
                if arm not in r or r["query_id"] not in gold:
                    continue
                n += 1
                old = int(bool(r[arm].get("answer_em")))
                new = int(hitab_exact_match_text(r[arm]["pred"], gold[r["query_id"]]))
                before += old
                after += new
                if old != new:
                    changed += 1
                    r[arm]["answer_em"] = new
            if n and before != after:
                per_arm[arm] = {"n": n, "before": round(before / n, 4),
                                "after": round(after / n, 4),
                                "delta": round((after - before) / n, 4)}
        if not changed:
            continue
        entry = {"corpus": corpus, "records": str(rf), "queries_flipped": changed,
                 "arms": per_arm}

        sj = Path(str(rf).replace("_records.jsonl", ".json"))
        if sj.exists():
            d = json.loads(sj.read_text())
            for arm, v in d.get("summary", {}).items():
                if "answer_em" in v:
                    hits = [r for r in recs if arm in r and r["query_id"] in gold]
                    if hits:
                        v["answer_em"] = round(
                            sum(r[arm]["answer_em"] for r in hits) / len(hits), 4)
            for key in list(d.get("paired_tests", {})):
                if not key.startswith("answer_em:"):
                    continue
                a, b = key.split(":", 1)[1].split("_vs_")
                pairs = [r for r in recs if a in r and b in r and r["query_id"] in gold]
                d["paired_tests"][key] = mcnemar(
                    [r[a]["answer_em"] for r in pairs],
                    [r[b]["answer_em"] for r in pairs])
            entry["summary_json"] = str(sj)
            if args.apply:
                sj.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
        if args.apply:
            rf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                  for r in recs))
        ledger["runs"][rf.name] = entry
        print(f"  {rf.name:52} {corpus:13} flipped {changed:4}  "
              + " ".join(f"{a}{v['delta']:+.4f}" for a, v in per_arm.items()))

    Path(args.out).write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    print(f"\n{len(ledger['runs'])} runs affected -> {args.out}"
          + ("" if args.apply else "   (dry run, nothing written)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
