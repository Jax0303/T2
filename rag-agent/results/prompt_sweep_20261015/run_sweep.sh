#!/usr/bin/env bash
# 2단계 전수 탐색(vLLM) — PREREG-2026-10-15-prompt-sweep.md. 인자: <모델 hf id> <출력 이름> [변형 목록(기본 factorial:XFBDCE)] [추가 인자…]
# 한 프로세스에서 변형마다 C → A 순으로 돌린다(모델 한 번 적재, --resume 으로 이어 쓴다).
set -uo pipefail
cd "$(dirname "$0")/../.."
M="$1"; N="$2"; V="${3:-factorial:XFBDCE}"; shift 3 2>/dev/null || shift $#; D=results/prompt_sweep_20261015
echo "[start] $(date -Is) $M $V"
CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model "$M" \
  --records $D/dev_C_records.jsonl,$D/dev_A_records.jsonl --arm C,A --variants "$V" --out-dir $D/sweep_$N --resume "$@"
echo "[exit $?] $(date -Is)"
