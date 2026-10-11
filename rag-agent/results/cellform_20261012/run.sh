#!/usr/bin/env bash
# PREREG-2026-10-12-cellform.md — 미리보기 300 → 나머지 582. [나] 는 길이로 나눠 5,400토큰 이하 묶음(128), 초과 한 문항씩.
#   setsid nohup bash results/cellform_20261012/run.sh > results/cellform_20261012/run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
D=results/cellform_20261012
RUN="flock /home/user/T2-1/rag-agent/.gpu.lock env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer --prompt original --resume"
step() { echo "[start] $(date -Is) $1"; shift; $RUN "$@"; echo "[end] $(date -Is)"; }
for part in preview rest; do
  lim=(); [ $part = preview ] && lim=(--limit 300)
  step "A $part" --records $D/A_chunk_md_records.jsonl --ids $D/ids_order.jsonl "${lim[@]}" --out-dir $D/A_chunk_md
  step "C $part" --records $D/C_ours_budget_records.jsonl --ids $D/ids_order.jsonl "${lim[@]}" --out-dir $D/C_ours_budget
  step "B $part short" --records $D/B_chunk_sent_records.jsonl --ids $D/ids_B_${part}_short.jsonl --out-dir $D/B_chunk_sent
  step "B $part long" --records $D/B_chunk_sent_records.jsonl --ids $D/ids_B_${part}_long.jsonl --batch-size 1 --out-dir $D/B_chunk_sent
done
echo "[all done] $(date -Is)"
