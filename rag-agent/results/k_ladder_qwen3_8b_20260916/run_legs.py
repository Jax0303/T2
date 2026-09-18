"""k-ladder answer legs, each in its OWN subprocess, retried on crash.

Qwen3.5-9B (2026-09-15/16) does not fit this 8GB card reliably: its 248k-vocab
embed+lm_head alone are ~4GB unquantized bf16, leaving no headroom once k5's
longer prompt grows the KV cache -- either ~15x slowdown or an outright CUDA
OOM. Switched to Qwen3-8B (152k vocab, already used on this hardware in the
MultiHiertt line).

The environment itself is flaky independent of model choice: dmesg shows WSL's
GPU passthrough (`dxgk: dxgkio_make_resident: Ioctl failed: -12`) failing under
Windows-host GPU memory pressure, which surfaced as `torch.AcceleratorError:
CUDA error: unknown error` mid-leg (k1 died at 750/991 under the OLD version of
scripts/answer_accuracy.py, which only wrote rows at the very end -- all 750
generations were lost, not recoverable). answer_accuracy.py now writes each row
immediately and supports --resume. A CUDA driver error can leave the process's
CUDA context poisoned, so retries here are a FRESH subprocess each time (not a
retry of aa.main() in one process) -- --resume then regenerates only what's
missing.

  cd rag-agent && PYTHONPATH=. .venv/bin/python results/k_ladder_qwen3_8b_20260916/run_legs.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = "results/k_ladder_qwen3_8b_20260916"
MAX_ATTEMPTS = 8

for leg in ("k1", "k5", "k10", "k20", "gold"):
    out = f"{D}/s3c_v2_primary_qwen3_8b_{leg}.jsonl"
    if (ROOT / out).with_suffix(".json").exists():
        print("skip", leg, flush=True)
        continue
    extra = ["--condition", "gold"] if leg == "gold" else ["--k", leg[1:]]
    cmd = [sys.executable, "scripts/answer_accuracy.py",
           "--records", "results/evaluation_v2/s3c_v2_records.jsonl",
           "--reader", "local:Qwen/Qwen3-8B?quantization=4bit", "--prompt", "neutral",
           "--max-tokens", "64", "--primary-only", "--resume", *extra, "--out", out]
    print("==", leg, flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode == 0:
            break
        print(f"[retry] {leg} attempt {attempt} exit={r.returncode}", flush=True)
        time.sleep(15)
    else:
        raise SystemExit(f"{leg}: {MAX_ATTEMPTS}번 재시도했지만 끝나지 않았다")
