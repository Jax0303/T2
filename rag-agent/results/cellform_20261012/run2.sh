#!/usr/bin/env bash
# PREREG-2026-10-12-cellform.md 실행 세부 2: [나] 는 길이와 상관없이 전부 한 문항씩(묶음 생성이 3k 토큰대 입력에서도 멈춤).
#   미리보기 [나] 300 → 나머지 [가]·[다] 582 → 나머지 [나] 582.
set -euo pipefail
cd "$(dirname "$0")/../.."
D=results/cellform_20261012
RUN="flock /home/user/T2-1/rag-agent/.gpu.lock env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer --prompt original --resume"
step() { echo "[start] $(date -Is) $1"; shift; $RUN "$@"; echo "[end] $(date -Is)"; }
step "B preview (bs1)" --records $D/B_chunk_sent_records.jsonl --ids $D/ids_order.jsonl --limit 300 --batch-size 1 --out-dir $D/B_chunk_sent
step "A rest" --records $D/A_chunk_md_records.jsonl --ids $D/ids_order.jsonl --out-dir $D/A_chunk_md
step "C rest" --records $D/C_ours_budget_records.jsonl --ids $D/ids_order.jsonl --out-dir $D/C_ours_budget
step "B rest (bs1)" --records $D/B_chunk_sent_records.jsonl --ids $D/ids_order.jsonl --batch-size 1 --out-dir $D/B_chunk_sent
echo "[all done] $(date -Is)"
