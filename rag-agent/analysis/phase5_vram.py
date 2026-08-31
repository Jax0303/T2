#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 5 Task A -- measured 4-bit VRAM for the candidate readers."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CANDS = ["Qwen/Qwen2.5-7B-Instruct",          # incumbent, for reference
         "Qwen/Qwen2.5-Math-7B-Instruct",
         "Qwen/Qwen2.5-Coder-7B-Instruct",
         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
Q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)
OUT = Path("results/phase5/vram.json")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["model"] for r in rows}
    for name in CANDS:
        if name in done:
            continue
        r = {"model": name}
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(name)
            m = AutoModelForCausalLM.from_pretrained(
                name, quantization_config=Q, device_map={"": 0},
                torch_dtype=torch.float16)
            r["load_sec"] = round(time.time() - t0, 1)
            r["revision"] = m.config._name_or_path and getattr(
                m.config, "_commit_hash", None)
            r["weights_MiB"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            # one short generation, so the peak includes a real forward pass
            ids = tok("CONTEXT: a | b | 1\n\nQUESTION: what is 1+1?\n\nAnswer:",
                      return_tensors="pt").to(0)
            with torch.no_grad():
                m.generate(**ids, max_new_tokens=32, do_sample=False)
            r["peak_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            r["ok"] = True
        except Exception as e:                                   # noqa: BLE001
            r["ok"] = False
            r["error"] = f"{type(e).__name__}: {e}"[:400]
        rows.append(r)
        OUT.write_text(json.dumps(rows, indent=2))
        print(json.dumps(r), flush=True)
        for v in ("m", "tok", "ids"):
            if v in dir():
                pass
        m = tok = ids = None
        gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
