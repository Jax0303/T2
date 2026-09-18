"""Controlled-QA legs (scripts/bottleneck_diagnosis.py item 3), each in its OWN
subprocess, retried on crash — same pattern and same reason as
results/k_ladder_qwen3_8b_20260916/run_legs.py: a CUDA driver error under this
card's WSL passthrough can poison the process's CUDA context, so a retry has to
be a fresh subprocess, not a retry inside one process. bottleneck_diagnosis.py
writes each row immediately and supports --resume, so a fresh subprocess only
regenerates what a killed attempt did not finish.

  cd rag-agent && PYTHONPATH=. .venv/bin/python results/bottleneck_diagnosis/run_legs.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = "results/bottleneck_diagnosis"
MAX_ATTEMPTS = 8

sys.path.insert(0, str(ROOT / "scripts"))
from bottleneck_diagnosis import condition_names  # noqa: E402

for cond in condition_names():
    out = f"{D}/controlled_qa_{cond}.jsonl"
    if (ROOT / out).with_suffix(".json").exists():
        print("skip", cond, flush=True)
        continue
    cmd = [sys.executable, "scripts/bottleneck_diagnosis.py", "controlledqa",
           "--condition", cond, "--resume", "--out", out]
    print("==", cond, flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode == 0:
            break
        print(f"[retry] {cond} attempt {attempt} exit={r.returncode}", flush=True)
        time.sleep(15)
    else:
        raise SystemExit(f"{cond}: {MAX_ATTEMPTS}번 재시도했지만 끝나지 않았다")

print("all controlled-QA legs done", flush=True)
