#!/usr/bin/env bash
# GPT-4o 확인 실험 4조건 — PREREG-2026-10-13-gpt4o.md. 두 사슬을 동시에: A = 표 전부 → 질문만, B = 본 방법 → LangChain.
#   setsid nohup bash results/gpt4o_20261013/run.sh A > results/gpt4o_20261013/run_A.log 2>&1 &
#   setsid nohup bash results/gpt4o_20261013/run.sh B > results/gpt4o_20261013/run_B.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
D=results/gpt4o_20261013
R="openai:gpt-4o?key_env=OPENAI_API_KEY_OLD"
run() {  # 이름 프롬프트 레코드
  echo "[start] $(date -Is) $1"
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/baselines/langchain_chunk.py \
    --stage answer --prompt "$2" --reader "$R" --batch-size 1 --records "$3" --out-dir "$D/ans_$1" --resume
}
case "$1" in
  A) run fulldoc original $D/fulldoc_records.jsonl; run closed closed $D/closed_records.jsonl ;;
  B) run ours original results/cellform_20261012/C_ours_budget_records.jsonl; run langchain original results/cellform_20261012/A_chunk_md_records.jsonl ;;
esac
echo "[done] $(date -Is) $1"
