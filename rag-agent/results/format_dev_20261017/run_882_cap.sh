#!/usr/bin/env bash
# 882 확인(표 수 제한 변형): 인자 <X 팔 이름(D1|D2|DF2)>. 팔 X·D·A × P*, 3,072 토큰, 실행 2회(동시 32 / 24). PREREG-2026-10-18-table-cap-882.md(결과 뒤 작성·커밋 후 실행)
set -uo pipefail
cd "$(dirname "$0")/../.."
X="$1"; D=results/format_dev_20261017; CF=results/cellform_20261012; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for RUN in 1 2; do
  SEQS=$([ "$RUN" = 1 ] && echo 32 || echo 24)
  echo "[start run$RUN seqs=$SEQS] $(date -Is)"
  env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len 7168 --max-num-seqs $SEQS \
    --records $D/${X}882_records.jsonl,$D/D882_records.jsonl,$CF/A_chunk_md_records.jsonl --arm $X,D,A \
    --variants "$V" --out-dir $D/final882cap_run$RUN --max-new-tokens 3072 --resume
  echo "[exit $? run$RUN] $(date -Is)"
done
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python $D/score_formats2.py $D/final882cap_run1 $D/final882cap_run2 --gold train | tee $D/score_882cap.txt
echo "[all done] $(date -Is)"
