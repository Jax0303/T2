#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-score a finished RealHiTBench run from its records, without an LLM.

`realhitbench_answer_accuracy.py` already re-scores strict from the stored
pred/gold on every run, but only inside the loop that calls the solver, so
changing the scorer would otherwise mean paying for the answers again. The
records hold the prediction and the gold, which is everything the strict scorer
needs, so this recomputes the strict blocks and rewrites the result file in
place. The lenient blocks are left as they were written: they come from a
different scorer whose per-record verdict is what was stored.

Run:
    PYTHONPATH=. .venv/bin/python scripts/rescore_realhitbench.py \
        results/realhitbench_s1_vs_s2_gpt51.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realhitbench_answer_accuracy import (em_norm, gold_is_numeric, mcnemar_p,
                                          token_f1)


def f1_block(rs) -> dict:
    """Token-level F1, the metric RealHiTBench reports next to EM (§5.1).

    Averaged per question, not pooled — every question weighs the same, which is
    what an average of per-item F1 means in that paper and in SQuAD before it.
    Paired significance is left out on purpose: McNemar is a test on binary
    flips, and applying it to a graded score would be a category error.
    """
    m = len(rs)
    if not m:
        return {"n": 0}
    b = sum(r["f1_base"] for r in rs) / m
    t = sum(r["f1_treat"] for r in rs) / m
    return {"n": m, "base": round(b, 4), "treat": round(t, 4),
            "delta": round(t - b, 4)}


def block(rs, strict: bool) -> dict:
    kb = "correct_base_strict" if strict else "correct_base"
    kt = "correct_treat_strict" if strict else "correct_treat"
    b = sum(r[kt] and not r[kb] for r in rs)
    c = sum(r[kb] and not r[kt] for r in rs)
    m = len(rs)
    return {"n": m,
            "base": round(sum(r[kb] for r in rs) / m, 4),
            "treat": round(sum(r[kt] for r in rs) / m, 4),
            "delta": round((sum(r[kt] for r in rs) - sum(r[kb] for r in rs)) / m, 4),
            "treat_only": b, "base_only": c,
            "mcnemar_p": round(float(mcnemar_p(b, c)), 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="results/realhitbench_*.json to rewrite")
    ap.add_argument("--records", default="",
                    help="defaults to <result stem>_records.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.result)
    recs_path = Path(args.records) if args.records else path.with_name(
        path.stem + "_records.jsonl")
    out = json.loads(path.read_text())
    recs = [json.loads(l) for l in recs_path.open()]
    if not recs:
        print(f"no records in {recs_path}")
        return 1

    changed = 0
    for r in recs:
        for arm in ("base", "treat"):
            new = bool(em_norm(r[f"pred_{arm}"], r["gold"][0]))
            changed += int(new != bool(r.get(f"correct_{arm}_strict")))
            r[f"correct_{arm}_strict"] = new
            r[f"f1_{arm}"] = round(token_f1(r[f"pred_{arm}"], r["gold"][0]), 4)
        r["gold_numeric"] = gold_is_numeric(r["gold"][0])

    num = [r for r in recs if r["gold_numeric"]]
    txt = [r for r in recs if not r["gold_numeric"]]

    before = out.get("answer_accuracy_strict", {})
    after = block(recs, strict=True)
    out["answer_accuracy_strict"] = after
    out["by_compstruccata_strict"] = {
        cata: block([r for r in recs if r["cata"] == cata], strict=True)
        for cata in sorted({r["cata"] for r in recs})}

    # RealHiTBench reports F1 alongside EM (arXiv:2506.13405 §5.1); EM alone is
    # the stricter half of the pair.
    out["answer_f1"] = f1_block(recs)
    out["answer_f1_numeric_gold"] = f1_block(num)

    # The aggregation subset filter admits questions whose gold is a name, date
    # or sentence. The solver prompt requires a number, so those score 0 in every
    # arm and carry no discordant pair — dropping them moves both accuracies and
    # leaves McNemar's p untouched, which is why the split is reportable rather
    # than a choice of the flattering population.
    out["answer_accuracy_strict_numeric_gold"] = block(num, strict=True) if num else {"n": 0}
    out["answer_accuracy_strict_text_gold"] = block(txt, strict=True) if txt else {"n": 0}
    out.setdefault("population", {})["n_gold_numeric"] = len(num)
    out["population"]["n_gold_text"] = len(txt)

    out["scorer_note"] = ("strict blocks re-scored offline from stored pred/gold "
                          "by scripts/rescore_realhitbench.py; em_norm compares "
                          "at the gold's own decimal places (half-up, matching how "
                          "the golds were rounded), not a relative tolerance. "
                          "answer_f1 = token F1, the metric the benchmark reports "
                          "beside EM. *_numeric_gold excludes questions whose gold "
                          "is not a number: unanswerable under a numeric-only "
                          "solver prompt, 0 in every arm, no discordant pairs.")
    # the run's own note still described the tolerance it was written under
    if isinstance(out.get("note"), str) and "rel_tol" in out["note"]:
        out["note"] = out["note"].replace(
            "rel_tol 1e-5", "rounded to the gold's own decimal places")

    print(f"{path.name}: {changed} per-record verdicts changed")
    for k in ("base", "treat", "delta", "mcnemar_p"):
        print(f"   {k:10} {before.get(k)} -> {after[k]}")
    ng = out["answer_accuracy_strict_numeric_gold"]
    if ng.get("n"):
        print(f"   numeric-gold n={ng['n']}: {ng['base']} -> {ng['treat']} "
              f"(delta {ng['delta']}, p {ng['mcnemar_p']})")
    print(f"   token F1  n={out['answer_f1']['n']}: {out['answer_f1']['base']} "
          f"-> {out['answer_f1']['treat']} (delta {out['answer_f1']['delta']})")
    if args.dry_run:
        print("(dry run, nothing written)")
        return 0
    path.write_text(json.dumps(out, indent=2))
    with recs_path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote -> {path} (+ {recs_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
