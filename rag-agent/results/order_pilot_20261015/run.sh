#!/usr/bin/env bash
# PREREG-2026-10-15-prompt-order.md — 200문항, Context-first, Qwen3-8B 4비트.
#   setsid nohup bash results/order_pilot_20261015/run.sh > results/order_pilot_20261015/run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
D=results/order_pilot_20261015
RUN="env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer --prompt original_cf --resume --ids $D/ids200.jsonl"
echo "[start] $(date -Is) C"; $RUN --records results/cellform_20261012/C_ours_budget_records.jsonl --out-dir $D/C_ours_cf; echo "[end] $(date -Is)"
echo "[start] $(date -Is) A"; $RUN --records results/cellform_20261012/A_chunk_md_records.jsonl --out-dir $D/A_chunk_cf; echo "[end] $(date -Is)"
echo "[all done] $(date -Is)"
