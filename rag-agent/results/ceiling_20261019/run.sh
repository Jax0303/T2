#!/usr/bin/env bash
# 전달 단계 상한 진단 O1·O2·O3 × P*, 882, 4,096 토큰, 1회 — PREREG-2026-10-19-delivery-ceiling.md
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/ceiling_20261019; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for a in O1 O2 O3; do [ -f $O/${a}_records.jsonl ] || { echo "[missing $O/${a}_records.jsonl] $(date -Is)"; exit 1; }; done
MML=$(.venv/bin/python -c "import json; m=max(json.loads(l)['n_tok_ctx'] for a in ('O1','O2','O3') for l in open('$O/'+a+'_records.jsonl')); print(8192 if m+4600<=8192 else (10240 if m+4600<=10240 else 12288))")
echo "[start seqs=24 max-model-len=$MML] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len $MML --max-num-seqs 24 \
  --records $O/O1_records.jsonl,$O/O2_records.jsonl,$O/O3_records.jsonl --arm O1,O2,O3 --variants "$V" --out-dir $O/run1 --max-new-tokens 4096 --resume
echo "[exit $?] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_formats2.py $O/run1 --gold train | tee $O/score.txt
echo "[all done] $(date -Is)"
