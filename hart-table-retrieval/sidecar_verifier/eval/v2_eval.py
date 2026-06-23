"""End-to-end evaluation of the v2 pipeline on HiTab.

Reports HiTab-style execution accuracy with the standardized matcher from
``answer_accuracy._match_all_modes`` (exact / relaxed-0.5pct / relaxed-1pct),
broken down per route (alpha / beta / gamma) AND per route_gold so we can see
which router decisions are paying off.

Usage:
  GROQ_API_KEY=...  python -m sidecar_verifier.eval.v2_eval \\
      --reader groq --max-queries 100

  # No API key — sanity test only (mock reader, every answer 'N/A')
  python -m sidecar_verifier.eval.v2_eval --reader mock --max-queries 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import (
    get_answer,
    get_query_from_sample,
    get_table_from_sample,
    get_table_id,
    load_hitab,
)
from sidecar_verifier.agent.pipeline_v2 import VerifierAgentV2, MockReader
from sidecar_verifier.agent.retriever import VectorRetriever
from sidecar_verifier.store.table_store import TableStore
from sidecar_verifier.eval.answer_accuracy import _match_all_modes, _TOLERANCES


def make_reader(name: str, model: str = ""):
    if name == "groq":
        from sidecar_verifier.agent.groq_reader import GroqAnswerer
        return GroqAnswerer(model=model) if model else GroqAnswerer()
    if name == "openrouter":
        from sidecar_verifier.agent.openrouter_reader import OpenRouterAnswerer
        return OpenRouterAnswerer(model=model) if model else OpenRouterAnswerer()
    if name == "cerebras":
        from sidecar_verifier.agent.cerebras_reader import CerebrasAnswerer
        return CerebrasAnswerer(model=model) if model else CerebrasAnswerer()
    if name == "mock":
        return MockReader()
    if name == "local":
        from sidecar_verifier.agent.answerer import LocalLLMAnswerer
        # Wrap to satisfy the v2 protocol
        class _LocalAdapter(LocalLLMAnswerer):
            def answer_full(self, q, rec): return self.answer(q, rec)
            def answer_subtable(self, q, sub, table_id="", title=""):
                from sidecar_verifier.agent.subtable import render_subtable_for_llm
                # Treat the sub-table block as the full table for the local reader
                class _MiniRec:
                    title = title
                    df = sub.df
                    table_id = table_id
                    def col_header_path(self, c): return [sub.df.columns[c]]
                    def row_header_path(self, r): return [sub.df.index[r]]
                    top_header_paths = [[c] for c in sub.df.columns]
                    left_header_paths = [[r] for r in sub.df.index]
                return self.answer(q, _MiniRec())
            def code_for_query(self, q, rec):
                # Local Qwen-3B can do this but quality is low; placeholder.
                from sidecar_verifier.agent.groq_reader import CodeResult
                return CodeResult(code="'N/A'", raw_output="(local-no-code)", table_id=rec.table_id)
        return _LocalAdapter()
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/d/hart_data/hitab/HiTab")
    p.add_argument("--chroma-dir", default="/mnt/d/hart_data/chroma_db")
    p.add_argument("--serializer", default="plain_markdown")
    p.add_argument("--max-queries", type=int, default=0,
                   help="0 = full dev split")
    p.add_argument("--w-verify", type=float, default=0.2)
    p.add_argument("--reader", default="groq",
                   choices=["groq", "openrouter", "cerebras", "mock", "local"])
    p.add_argument("--reader-model", default="",
                   help="Override reader's default model id.")
    p.add_argument("--oracle-retrieval", action="store_true",
                   help="Skip retrieval and feed gold table directly.")
    p.add_argument("--out", default="results/v2_eval.json")
    args = p.parse_args()

    print("Loading HiTab dev ...")
    samples = load_hitab(data_dir=args.data_dir, split="dev")
    if args.max_queries:
        samples = samples[: args.max_queries]
    print(f"  {len(samples)} queries")

    print("Building TableStore ...")
    store = TableStore()
    seen = set()
    for s in load_hitab(data_dir=args.data_dir, split="dev"):
        if "table" not in s:
            continue
        tid = get_table_id(s)
        if tid in seen:
            continue
        seen.add(tid)
        t = get_table_from_sample(s)
        t["table_id"] = tid
        store.add(t)
    print(f"  TableStore: {len(store)} tables")

    print("Building VectorRetriever ...")
    retriever = VectorRetriever(chroma_dir=args.chroma_dir, serializer=args.serializer)

    print(f"Reader: {args.reader} (model={args.reader_model or 'default'})")
    reader = make_reader(args.reader, args.reader_model)
    agent = VerifierAgentV2(retriever, store, reader, w_verify=args.w_verify)

    # ---- evaluation loop ----
    rows = []
    correct_by_mode = {m: 0 for m in _TOLERANCES}
    correct_by_mode_route = defaultdict(lambda: defaultdict(int))
    route_count = Counter()
    route_gold_count = Counter()
    route_confusion = Counter()
    answer_source_count = Counter()
    n = 0
    n_ret_hit = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        q = get_query_from_sample(s)
        gold_table = get_table_id(s)
        gold_ans = get_answer(s)
        gold_formulas = s.get("answer_formulas")
        if not q or not gold_table:
            continue
        n += 1

        forced = gold_table if args.oracle_retrieval else None
        res = agent.run(q, gold_formulas=gold_formulas, gold_answer=gold_ans,
                        forced_top_table=forced)
        ret_hit = res.top_table_id == gold_table
        n_ret_hit += int(ret_hit)

        modes = _match_all_modes(res.answer, gold_ans)
        for mode, (ok, _) in modes.items():
            correct_by_mode[mode] += int(ok)
            correct_by_mode_route[res.route][mode] += int(ok)

        route_count[res.route] += 1
        if res.route_gold:
            route_gold_count[res.route_gold] += 1
            route_confusion[(res.route_gold, res.route)] += 1
        answer_source_count[res.answer_source] += 1

        rows.append({
            "query": q,
            "gold_table": gold_table,
            "top_table": res.top_table_id,
            "retrieval_hit": ret_hit,
            "route_pred": res.route,
            "route_gold": res.route_gold,
            "answer_source": res.answer_source,
            "answer": res.answer,
            "gold_answer": gold_ans,
            "exact": modes["exact"][0],
            "exact_match_type": modes["exact"][1],
            "relaxed_0.5pct": modes["relaxed-0.5pct"][0],
            "relaxed_1pct": modes["relaxed-1pct"][0],
            "code": res.code,
            "sandbox_error": res.sandbox_error,
            "grounded_fraction": res.trace.grounded_fraction if res.trace else None,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] {(time.time()-t0):.0f}s elapsed")

    elapsed = time.time() - t0

    summary = {
        "n": n,
        "elapsed_sec": elapsed,
        "retrieval_R@1": n_ret_hit / n if n else 0,
        "answer_accuracy": {m: correct_by_mode[m] / n for m in _TOLERANCES},
        "per_route_count_pred": dict(route_count),
        "per_route_count_gold": dict(route_gold_count),
        "per_route_exact_acc": {
            r: correct_by_mode_route[r]["exact"] / max(route_count[r], 1)
            for r in route_count
        },
        "answer_source_distribution": dict(answer_source_count),
        "route_confusion": {f"gold={g}__pred={p}": v
                            for (g, p), v in route_confusion.items()},
        "router_accuracy_vs_gold":
            sum(v for (g, p), v in route_confusion.items() if g == p) /
            max(sum(route_confusion.values()), 1),
    }

    print("\n=== Summary ===")
    print(f"queries: {n}, elapsed: {elapsed:.0f}s")
    print(f"retrieval R@1: {summary['retrieval_R@1']:.3f}")
    print(f"answer accuracy:")
    for m, acc in summary["answer_accuracy"].items():
        print(f"  {m:<16} {acc:.3f}")
    print(f"router accuracy vs gold: {summary['router_accuracy_vs_gold']:.3f}")
    print(f"route_count (predicted): {dict(route_count)}")
    print(f"route_count (gold):      {dict(route_gold_count)}")
    print(f"per-route exact acc: {summary['per_route_exact_acc']}")
    print(f"answer source distribution: {dict(answer_source_count)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary["rows"] = rows
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
