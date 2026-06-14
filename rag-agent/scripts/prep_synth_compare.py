#!/usr/bin/env python3
"""Head-to-head: LLM-generated vs template C3 questions, same corpus.

The preprocessing ladder (prep_retrieval_eval.py) measures C3 = "+ synthetic
questions appended to the indexed text". Whether those questions are written by
a deterministic template or by an LLM is a knob (--synth). This script compares
two prep runs that were produced on the *identical* reduced corpus and query
sample (same --dataset/--split/--seed/--n-queries/--max-corpus) but different
--synth providers, and reports the paired delta of the LLM run minus the
template run at C3.

Because both runs share the seed, their query sets are identical; we align the
per-query gold ranks by question_id and run the same paired bootstrap used
elsewhere in the ladder. The quantity of interest is "what does an LLM buy over
free templates", isolated from corpus difficulty.

Usage
-----
python rag-agent/scripts/prep_synth_compare.py \
    --template results/prep/owt_bm25_cap1k_template.json \
    --llm      results/prep/owt_bm25_cap1k_llm.json \
    --cond C3 --ks 1,5,10 \
    --out results/prep/owt_bm25_cap1k_synth_compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.prep.stats import paired_delta_bootstrap, summarize_condition  # noqa: E402


def load(path: str, cond: str):
    d = json.loads(Path(path).read_text())
    qids = d["per_query"]["question_ids"]
    if cond not in d["per_query"]:
        raise SystemExit(f"{path}: condition {cond!r} not in per_query "
                         f"(have {list(d['per_query'])})")
    ranks = d["per_query"][cond]
    return d, dict(zip(qids, ranks))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", required=True, help="prep run with --synth template")
    p.add_argument("--llm", required=True, help="prep run with --synth llm:<spec>")
    p.add_argument("--cond", default="C3", help="condition to compare (default C3)")
    p.add_argument("--ks", default="1,5,10")
    p.add_argument("--boot-iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    d_t, ranks_t = load(args.template, args.cond)
    d_l, ranks_l = load(args.llm, args.cond)

    # align by question_id; require identical query sets so the pairing is exact
    common = [q for q in d_t["per_query"]["question_ids"] if q in ranks_l]
    if set(d_t["per_query"]["question_ids"]) != set(d_l["per_query"]["question_ids"]):
        print(f"WARNING: query sets differ; comparing {len(common)} shared queries "
              f"(template {len(ranks_t)} / llm {len(ranks_l)})", file=sys.stderr)
    a = [ranks_l[q] for q in common]   # LLM
    b = [ranks_t[q] for q in common]   # template

    result = {
        "template_run": args.template,
        "llm_run": args.llm,
        "cond": args.cond,
        "n_queries": len(common),
        "llm_synth_spec": d_l.get("config", {}).get("synth"),
        "template_synth_spec": d_t.get("config", {}).get("synth"),
        "n_tables": d_l.get("n_tables"),
        "summary": {
            "template": summarize_condition(b, ks),
            "llm": summarize_condition(a, ks),
        },
        "delta_llm_minus_template": {},
    }

    print(f"cond {args.cond} | {len(common)} paired queries | "
          f"corpus {d_l.get('n_tables')} tables")
    print(f"{'metric':8} {'template':>9} {'llm':>9} {'Δ(llm-tmpl)':>14} {'95% CI':>22}")
    for k in ks:
        mean, lo, hi = paired_delta_bootstrap(a, b, k, n_iters=args.boot_iters,
                                              seed=args.seed)
        sig = lo > 0 or hi < 0
        rt = summarize_condition(b, [k])[f"R@{k}"]
        rl = summarize_condition(a, [k])[f"R@{k}"]
        result["delta_llm_minus_template"][f"R@{k}"] = {
            "template": rt, "llm": rl,
            "delta": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": sig,
        }
        mark = " *" if sig else ""
        print(f"R@{k:<6} {rt:>9.4f} {rl:>9.4f} {mean:>+14.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]{mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
