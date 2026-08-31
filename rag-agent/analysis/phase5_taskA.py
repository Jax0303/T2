#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 5 Task A -- 4-bit VRAM and a 3-query hitab_arith smoke run for each
candidate reader. Same prompt, decoding and gold_cell context as Phase 4."""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import torch                                                         # noqa: E402
from baseline_comparison_llm import Budget                           # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from qwen_equiv_k import SYS, args_for                               # noqa: E402
from rag_agent.llm.local_qwen import LocalQwenLLM                    # noqa: E402
from stratified_recall import build_chunks                           # noqa: E402

# PREREGISTER 개정 6: DeepSeek-R1-Distill-Qwen-7B 제외
CANDS = ["Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"]
SEED, MAXNEW, NSMOKE = 42, 32, 3
OUT = Path("results/phase5/taskA.json")


def smoke_jobs(n=NSMOKE):
    """The first 3 hitab_arith gold_cell rows of Phase 4, by query_id."""
    rows = [json.loads(l) for l in open("results/phase4/reader_records.jsonl")]
    g = sorted((r for r in rows if r["pool"] == "hitab_arith"
                and r["policy"] == "gold_cell"), key=lambda r: r["query_id"])
    C = load_corpus(args_for("hitab", "hitab_dev_lookup_all"))
    by = defaultdict(dict)
    for k, (t, i, j) in enumerate(C.cell_owner):   # not n: n is the slice size
        by[t][(i, j)] = k
    # same S3c cell sentence Phase 4's gold_cell condition injected
    text = {}
    chunks, owner = build_chunks(C, "P4_path_cell",
                                 Budget("BAAI/bge-small-en-v1.5"), by, "S3c")
    for c, (tid, cells) in zip(chunks, owner):
        for (i, j) in cells:
            text[(tid, i, j)] = c.text
    jobs = []
    for r in g[:n]:
        cells = [tuple(c) for c in json.loads(r["gold_cell"])]
        jobs.append(dict(query_id=r["query_id"], query=r["query"],
                         gold_answer=r["gold_answer"],
                         phase4_raw=r["pred_answer_raw"],
                         ctx="\n".join(text[c] for c in cells)))
    return jobs


def run(llm, job):
    tok = llm.tokenizer
    user = f"CONTEXT:\n{job['ctx']}\n\nQUESTION: {job['query']}\n\nAnswer:"
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ins = tok(prompt, return_tensors="pt").to(llm.device)
    t0 = time.time()
    with torch.inference_mode():
        out = llm.model.generate(**ins, max_new_tokens=MAXNEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    gen = out[0, ins["input_ids"].shape[1]:]
    ids = gen.tolist()
    eos = tok.eos_token_id
    stopped = any(i in {eos, tok.convert_tokens_to_ids("<|im_end|>")} for i in ids)
    return dict(query_id=job["query_id"], query=job["query"],
                gold_answer=job["gold_answer"], phase4_raw=job["phase4_raw"],
                prompt_tokens=int(ins["input_ids"].shape[1]),
                n_generated=len(ids), hit_cap=len(ids) >= MAXNEW,
                stopped_on_eos=bool(stopped),
                raw=tok.decode(gen, skip_special_tokens=True),
                raw_with_specials=tok.decode(gen, skip_special_tokens=False),
                latency_sec=round(time.time() - t0, 3))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["model"] for r in rows}
    jobs = smoke_jobs()
    torch.manual_seed(SEED)
    for name in CANDS:
        if name in done:
            continue
        r = {"model": name}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            llm = LocalQwenLLM(model_name=name, dtype="float16",
                               quantization="4bit")
            r["load_sec"] = round(time.time() - t0, 1)
            r["revision"] = getattr(llm.model.config, "_commit_hash", None)
            r["weights_MiB"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
            r["smoke"] = [run(llm, j) for j in jobs]
            r["peak_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            r["ok"] = True
            del llm
        except Exception as e:                                       # noqa: BLE001
            r["ok"] = False
            r["error"] = f"{type(e).__name__}: {e}"[:500]
        rows.append(r)
        OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in r.items() if k != "smoke"}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
