#!/usr/bin/env bash
# vLLM 결정적 생성 확인: 같은 60문항(dev 앞 60, 본 방법 목록 C, P*, 1,536)을 (a) VLLM_BATCH_INVARIANT=1·동시 32 (b) VLLM_BATCH_INVARIANT=1·동시 8 (c) 기본·동시 8 로 돌려 글자 단위 비교.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/determinism_20261018; SW=results/prompt_sweep_20261015; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
run() { # name, extra env, max-num-seqs
  echo "[start $1] $(date -Is)"
  env $ENV $2 ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 6144 --max-num-seqs $3 \
    --records $SW/dev_C_records.jsonl --arm C --variants "$V" --out-dir $D/$1 --max-new-tokens 1536 --ids $D/ids60.json --resume
  echo "[exit $? $1] $(date -Is)"
}
run inv_s32 "VLLM_BATCH_INVARIANT=1" 32
run inv_s8  "VLLM_BATCH_INVARIANT=1" 8
run def_s8  "" 8
echo "[all done] $(date -Is)"
