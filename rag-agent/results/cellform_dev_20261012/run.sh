#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
D=results/cellform_dev_20261012
for c in A C K1 K2 K3 K4; do
  echo "[start] $(date -Is) $c"
  flock /home/user/T2-1/rag-agent/.gpu.lock env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/baselines/langchain_chunk.py \
    --stage answer --prompt original --split validation --records $D/${c}_records.jsonl --ids results/reader_format_20260927/mh_dev_pop332.jsonl --out-dir $D/ans_$c
done
echo "[all done] $(date -Is)"
