#!/usr/bin/env python3
"""Remeasure the three reader legs invalidated by the 2026-09-10 audit.
No prompt search, training, retrieval rerun, or input truncation.
"""
from __future__ import annotations
import hashlib, json, sys, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.answer_accuracy import BASE, summarize
from rag_agent.eval.metrics import hitab_exact_match_text
from rag_agent.llm.factory import build_llm
D = ROOT / "results/audit_fix_20260910"
OUT = ROOT / "results/audit_answers_20260910"
READER = "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
ARMS = ("t_rowcol_values", "t_tablerag_leaf", "t_tablerag_path")

def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def context_sha(ctx):
    return hashlib.md5("\n".join(ctx).encode()).hexdigest()[:12]

def user_prompt(r):
    return "Context:\n" + "\n".join(r["context"]) + f"\n\nQuestion: {r['question']}\nAnswer:"

def checked_rows(path):
    rows = [r for r in read_rows(path) if "correct" in r]
    assert len(rows) == len({r["query_id"] for r in rows}) == 1581
    for r in rows:
        assert r["context_sha"] == context_sha(r["context"])
    return rows

def main():
    import torch, transformers, bitsandbytes
    OUT.mkdir(exist_ok=True)
    inputs = {tag: checked_rows(D / (tag + "_fix_records.jsonl")) for tag in ARMS}
    config = {
        "reader": READER, "prompt": BASE, "max_new_tokens": 64,
        "memory_policy": "empty_unused_cuda_cache_after_each_query",
        "prior_segment": json.loads((OUT/"RESOURCE_RESTART.json").read_text()),
        "temperature": 0, "seed": 42, "batch_size": 1,
        "source_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "runner_sha256": sha(__file__),
        "reader_sha256": sha(ROOT/"rag_agent/llm/local_qwen.py"),
        "scorer_sha256": sha(ROOT/"rag_agent/eval/metrics.py"),
        "answer_script_sha256": sha(ROOT/"scripts/answer_accuracy.py"),
        "protocol_sha256": sha(ROOT/"REMEASURE-2026-09-10-audit.md"),
        "sources": {tag: sha(D/(tag+"_fix_records.jsonl")) for tag in ARMS},
        "torch": torch.__version__, "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__, "cuda": torch.version.cuda,
    }
    meta = OUT/"RUN_CONFIG.json"
    if meta.exists():
        previous = json.loads(meta.read_text())
        if previous != config:
            assert previous == config["prior_segment"]["old_config"], "unrecorded migration"
            assert all(config[k] == v for k,v in previous.items() if k != "runner_sha256"), "inference setting changed"
            meta.write_text(json.dumps(config,indent=2))
    else:
        meta.write_text(json.dumps(config,indent=2))
    llm = build_llm(READER)
    torch.manual_seed(42)
    runtime = {
        "model_commit": getattr(llm.model.config,"_commit_hash",None),
        "gpu": torch.cuda.get_device_name(0),
        "attn": getattr(llm.model.config,"_attn_implementation",None),
        "context_limit": llm.context_limit,
        "device_map": {k:str(v) for k,v in getattr(llm.model,"hf_device_map",{}).items()},
    }
    (OUT/"RUNTIME.json").write_text(json.dumps(runtime,indent=2))
    # Exact baseline check; if the runtime changed predictions, stop before comparing arms.
    smoke_file = OUT/"BASELINE_CHECK_memory.json"
    if not smoke_file.exists():
        baseline = {r["query_id"]:r for r in read_rows(ROOT/"results/retrieval_accuracy/t_s3c_hybrid_answer_retrieved.jsonl")}
        reference = checked_rows(D/"t_s3c_hybrid_fix_records.jsonl")[:50]
        checks=[]
        for r in reference:
            pred=llm.complete(BASE,user_prompt(r),max_tokens=64,temperature=0.)
            torch.cuda.empty_cache()
            old=baseline[r["query_id"]]
            checks.append({"query_id":r["query_id"],"pred":pred,"old_pred":old["pred"],
                           "identical":pred==old["pred"] and int(hitab_exact_match_text(pred,r["answer"]))==old["answer_correct"]})
        smoke_file.write_text(json.dumps({"n":50,"all_identical":all(x["identical"] for x in checks),"rows":checks},indent=2))
    assert json.loads(smoke_file.read_text())["all_identical"], "baseline runtime drift; fresh baseline required"
    print("baseline 50/50 predictions and scores identical",flush=True)
    # Token counts are measured without changing or truncating input.
    token_stats={}
    for tag,rows in inputs.items():
        toks=[llm.n_prompt_tokens(BASE,user_prompt(r)) for r in rows]
        assert max(toks)+64 <= llm.context_limit
        token_stats[tag]={"n":len(toks),"mean":sum(toks)/len(toks),"max":max(toks),
                          "over_8192":sum(t>8192 for t in toks),"over_context_limit":0}
    (OUT/"TOKEN_AUDIT.json").write_text(json.dumps(token_stats,indent=2))
    print(json.dumps(token_stats),flush=True)
    for tag,rows in inputs.items():
        out=OUT/(tag+"_answer_retrieved.jsonl")
        done=read_rows(out) if out.exists() else []
        assert len(done)==len({r["query_id"] for r in done})
        assert [r["query_id"] for r in done]==[r["query_id"] for r in rows[:len(done)]]
        for a,r in zip(done,rows):
            assert a["context_sha"]==r["context_sha"]
        t0=time.time()
        with out.open("a") as f:
            for index,r in enumerate(rows[len(done):],len(done)+1):
                prompt=user_prompt(r)
                n_tok=llm.n_prompt_tokens(BASE,prompt)
                t=time.time()
                pred=llm.complete(BASE,prompt,max_tokens=64,temperature=0.)
                allocated = torch.cuda.memory_allocated()/2**20
                reserved = torch.cuda.memory_reserved()/2**20
                torch.cuda.empty_cache()
                row={"gpu_allocated_mb":allocated,"gpu_reserved_before_clear_mb":reserved,
                     "query_id":r["query_id"],"mode":r["mode"],
                     "retrieval_correct":r["correct"],"answer_correct":int(hitab_exact_match_text(pred,r["answer"])),
                     "aggregation":r.get("aggregation"),"n_ctx":len(r["context"]),
                     "context_sha":r["context_sha"],"n_tok":n_tok,
                     "cells_in_context":r["cells_in_context"],"pred":pred,"answer":r["answer"],
                     "seconds":time.time()-t}
                f.write(json.dumps(row,ensure_ascii=False)+"\n"); f.flush()
                done.append(row)
                if index%50==0 or index==len(rows):
                    print(json.dumps({"arm":tag,"done":index,"total":len(rows),"seconds":round(time.time()-t0)}),flush=True)
        summary={"records":str((D/(tag+"_fix_records.jsonl")).relative_to(ROOT)),
                 "source_sha256":config["sources"][tag],"complete":True,"condition":"retrieved",
                 "reader":llm.name,"prompt":"base","seed":42,"max_new_tokens":64,"batch_size":1,
                 "excluded_unit_defect":False,**summarize(done,llm.context_limit)}
        out.with_suffix(".json").write_text(json.dumps(summary,indent=2))
        print(json.dumps({"arm":tag,"complete":True,"n":len(done)}),flush=True)

if __name__=="__main__":
    main()
