#!/usr/bin/env python3
"""Fixed-reader, fixed-retrieval comparison of lossless context layouts."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_agent.serialization.sparse_context import build_visible_lookup, render_context

D = ROOT / "results/retrieval_accuracy"
OUT = ROOT / "results/sparse_context_v1"
READER = "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def scored(template):
    tag = "t_s3c_hybrid" if template == "s3c" else "t_mt2net_hybrid"
    path = D / f"{tag}_records.jsonl"
    return path, [r for r in read_jsonl(path) if "correct" in r]

def lookup_for(template):
    _, rs = scored(template)
    tids = {r["table_id"] for r in read_jsonl(D / "t_s3c_hybrid_records.jsonl")}
    page_titles = json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text())
    return build_visible_lookup(str(ROOT / "data/hitab"), tids, template, page_titles)

def validate():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"prereg_sha256": sha(ROOT / "PREREG-2026-09-09-sparse-context.md"), "templates": {}}
    for template in ("s3c", "mt2net"):
        path, rows = scored(template)
        lookup = lookup_for(template)
        stats = Counter()
        preview = []
        for r in rows:
            assert len(r["context"]) == r["cells_in_context"] == 20
            for layout in ("original", "sparse", "grouped"):
                text, info = render_context(r["context"], lookup, layout)
                assert info["n_preserved_lines"] == len(r["context"])
                if layout == "original":
                    assert text == "\n".join(r["context"])
                stats[layout + "_queries"] += 1
                stats[layout + "_fallback_lines"] += info["fallback"]
            if len(preview) < 3:
                rendered, _ = render_context(r["context"], lookup, "sparse")
                preview.append({"query_id": r["query_id"], "question": r["question"], "rendered": rendered})
        report["templates"][template] = {"source_sha256": sha(path), "n": len(rows), **dict(stats)}
        (OUT / f"{template}_preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    (OUT / "PREFLIGHT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

def run(template, layout, smoke=False):
    import torch
    import transformers
    import bitsandbytes
    from rag_agent.llm.factory import build_llm
    from rag_agent.eval.metrics import hitab_exact_match_text
    from scripts.answer_accuracy import BASE
    path, rows = scored(template)
    if smoke:
        if (template, layout) != ("s3c", "original"):
            raise ValueError("smoke is the frozen original baseline only")
        rows = rows[:50]
    out = OUT / (f"{template}_{layout}" + ("_smoke" if smoke else "") + ".jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_suffix(".meta.json")
    lookup = {} if layout == "original" else lookup_for(template)
    config = {
        "source": str(path.relative_to(ROOT)), "source_sha256": sha(path),
        "renderer_sha256": sha(ROOT / "rag_agent/serialization/sparse_context.py"),
        "runner_sha256": sha(__file__),
        "reader_backend_sha256": sha(ROOT / "rag_agent/llm/local_qwen.py"),
        "scorer_sha256": sha(ROOT / "rag_agent/eval/metrics.py"),
        "prereg_sha256": sha(ROOT / "PREREG-2026-09-09-sparse-context.md"),
        "reader": READER, "layout": layout, "template": template,
        "seed": 42, "max_new_tokens": 64, "temperature": 0, "batch_size": 1,
        "prompt": BASE, "n_expected": len(rows), "smoke": smoke,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
    }
    if meta_path.exists():
        assert json.loads(meta_path.read_text()) == config, "resume config differs"
    else:
        with meta_path.open("x") as f:
            json.dump(config, f, indent=2)
    done_rows = read_jsonl(out) if out.exists() else []
    assert len({r["query_id"] for r in done_rows}) == len(done_rows)
    assert [r["query_id"] for r in done_rows] == [r["query_id"] for r in rows[:len(done_rows)]]
    if len(done_rows) == len(rows):
        print(f"already complete: {out}", flush=True)
        return
    llm = build_llm(READER)
    torch.manual_seed(42)
    model_runtime = {
        "name": llm.name,
        "commit_hash": getattr(llm.model.config, "_commit_hash", None),
        "device_map": {k: str(v) for k, v in getattr(llm.model, "hf_device_map", {}).items()},
        "attn_implementation": getattr(llm.model.config, "_attn_implementation", None),
        "gpu": torch.cuda.get_device_name(0),
    }
    out.with_suffix(".runtime.json").write_text(json.dumps(model_runtime, indent=2))
    print(json.dumps(model_runtime), flush=True)
    primary = {r["query_id"] for r in scored("s3c")[1]
               if r["mode"] == "all" and (r.get("aggregation") or "none") == "none" and r["m"] == 1}
    t0 = time.time()
    with out.open("a") as f:
        for index, r in enumerate(rows[len(done_rows):], len(done_rows)+1):
            context, audit = render_context(r["context"], lookup, layout)
            user = "Context:\n" + context + f"\n\nQuestion: {r['question']}\nAnswer:"
            n_tok = llm.n_prompt_tokens(BASE, user)
            assert n_tok + 64 <= llm.context_limit, "context window exceeded"
            t = time.time()
            pred = llm.complete(BASE, user, max_tokens=64, temperature=0.0)
            row = {
                "query_id": r["query_id"], "pred": pred, "answer": r["answer"],
                "answer_correct": int(hitab_exact_match_text(pred, r["answer"])),
                "retrieval_correct": r["correct"], "mode": r["mode"],
                "aggregation": r.get("aggregation"), "m": r["m"],
                "n_tok": n_tok, "n_cells": r["cells_in_context"],
                "input_sha256": hashlib.sha256((BASE + "\0" + user).encode()).hexdigest(),
                "seconds": time.time()-t, "audit": audit,
            }
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
            f.flush()
            done_rows.append(row)
            if index % 25 == 0 or index == len(rows):
                pr = [x for x in done_rows if x["query_id"] in primary]
                print(json.dumps({"layout": layout, "template": template, "done": index,
                                  "total": len(rows), "elapsed_s": round(time.time()-t0),
                                  "primary_n_so_far": len(pr),
                                  "primary_correct_so_far": sum(x["answer_correct"] for x in pr)}), flush=True)
    summary = {**config, "complete": True, "n": len(done_rows),
               "answer_accuracy": sum(x["answer_correct"] for x in done_rows)/len(done_rows),
               "tokens_mean": sum(x["n_tok"] for x in done_rows)/len(done_rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    if smoke:
        archived = {x["query_id"]: x for x in read_jsonl(D / "t_s3c_hybrid_answer_retrieved.jsonl")}
        differences = [{"query_id": x["query_id"], "new": x["pred"], "old": archived[x["query_id"]]["pred"],
                        "new_correct": x["answer_correct"], "old_correct": archived[x["query_id"]]["answer_correct"]}
                       for x in done_rows if x["pred"] != archived[x["query_id"]]["pred"]
                       or x["answer_correct"] != archived[x["query_id"]]["answer_correct"]]
        check = {"n": len(done_rows), "identical_predictions_and_scores": not differences, "differences": differences}
        (OUT / "BASELINE_CHECK.json").write_text(json.dumps(check, indent=2))
        print(json.dumps(check), flush=True)

def compare():
    import numpy as np
    from scipy.stats import binomtest
    _, reference = scored("s3c")
    ids = {r["query_id"] for r in reference if r["mode"]=="all" and
           (r.get("aggregation") or "none")=="none" and r["m"]==1}
    smoke_path = OUT / "BASELINE_CHECK.json"
    archived_ok = smoke_path.exists() and json.loads(smoke_path.read_text())["identical_predictions_and_scores"]
    all_expected = {r["query_id"] for r in reference}
    results = {}
    for template in ("s3c", "mt2net"):
        fresh = OUT / f"{template}_original.jsonl"
        if fresh.exists() and len(read_jsonl(fresh)) == len(all_expected):
            base_path = fresh
        elif archived_ok:
            tag = "t_s3c_hybrid" if template=="s3c" else "t_mt2net_hybrid"
            base_path = D / f"{tag}_answer_retrieved.jsonl"
        else:
            continue
        base = {r["query_id"]: r for r in read_jsonl(base_path)}
        for layout in ("sparse", "grouped"):
            file = OUT / f"{template}_{layout}.jsonl"
            if not file.exists():
                continue
            new = {r["query_id"]: r for r in read_jsonl(file)}
            assert new.keys() <= all_expected
            if new.keys() != all_expected:
                results[f"{template}_{layout}"] = {"complete":False, "n_done":len(new), "n_expected":len(all_expected)}
                continue
            assert base.keys() == new.keys()
            ks = sorted(ids)
            a = np.array([base[k]["answer_correct"] for k in ks],dtype=np.int8)
            b = np.array([new[k]["answer_correct"] for k in ks],dtype=np.int8)
            gain, loss = int(((a==0)&(b==1)).sum()), int(((a==1)&(b==0)).sum())
            p = float(binomtest(min(gain,loss),gain+loss).pvalue) if gain+loss else 1.0
            delta = b-a
            rng=np.random.default_rng(42)
            ci=np.quantile(delta[rng.integers(0,len(ks),(10000,len(ks)))].mean(axis=1),[.025,.975]).tolist()
            groups = {}
            for mode in ("lookup_single","lookup_multi","arithmetic","header_answer","all_data","all_scored"):
                subset=[r["query_id"] for r in reference if
                        (mode=="lookup_single" and r["query_id"] in ids) or
                        (mode=="lookup_multi" and r["mode"]=="all" and (r.get("aggregation")or"none")=="none" and r["m"]>1) or
                        (mode=="arithmetic" and r["mode"]=="all" and (r.get("aggregation")or"none")!="none") or
                        (mode=="header_answer" and r["mode"]=="any") or
                        (mode=="all_data" and r["mode"]=="all") or mode=="all_scored"]
                groups[mode]={"n":len(subset), "base_correct":sum(base[k]["answer_correct"] for k in subset),
                              "new_correct":sum(new[k]["answer_correct"] for k in subset)}
            results[f"{template}_{layout}"] = {
                "complete":True, "baseline_source":str(base_path), "source":str(file),
                "n_primary":len(ids),"base_correct":int(a.sum()),"new_correct":int(b.sum()),
                "base_em":float(a.mean()),"new_em":float(b.mean()),"delta":float(delta.mean()),
                "gain":gain,"loss":loss,"mcnemar_p":p,"delta_ci95":ci,
                "target_0_8_reached":bool(b.sum()>=793),
                "improvement_rule_met_unadjusted":bool(delta.mean()>=.02 and p<.05),
                "groups":groups}
    (OUT/"COMPARISON.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=["validate","run","compare"])
    ap.add_argument("--template",choices=["s3c","mt2net"],default="s3c")
    ap.add_argument("--layout",choices=["original","sparse","grouped"],default="sparse")
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    if a.action=="validate":validate()
    elif a.action=="compare":compare()
    else:run(a.template,a.layout,a.smoke)

if __name__=="__main__":
    main()
