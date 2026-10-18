#!/usr/bin/env bash
# 생각 모드 dev 395(A·D, P*, 8,192) — PREREG-2026-10-19-thinking-dev.md. 60문항 측정에서 끝난 비율 ≥ 80% 일 때만 돈다.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/format_dev_20261017; SW=results/prompt_sweep_20261015; O=results/think_pilot_20261019; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
FIN=$(python3 - <<'PY'
import json
n=f=0
for a in ("D","A"):
    try:
        for l in open(f"results/think_pilot_20261019/think8192/{a}/R1X1F1B2D1C1E0.jsonl"):
            r=json.loads(l); n+=1; f+=int(r["finished"])
    except FileNotFoundError: pass
print(round(f/n,3) if n else 0)
PY
)
echo "[pilot finished ratio] $FIN $(date -Is)"
if python3 -c "import sys; sys.exit(0 if float('$FIN')>=0.8 else 1)"; then
  echo "[start dev think] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 12288 --max-num-seqs 16 \
    --records $SW/dev_A_records.jsonl,$D/D_records.jsonl --arm A,D --variants "$V" --out-dir $O/dev_think8192 --max-new-tokens 8192 --thinking --resume
  echo "[exit $?] $(date -Is)"
  env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_formats2.py $O/dev_think8192 | tee $O/score_dev_think.txt
else
  echo "[skip dev think: finished ratio $FIN < 0.8] $(date -Is)"
fi
echo "[all done] $(date -Is)"
