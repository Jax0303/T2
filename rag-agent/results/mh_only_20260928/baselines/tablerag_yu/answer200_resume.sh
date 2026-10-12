#!/usr/bin/env bash
# 2026-09-29 13:4x: 답변이 GPU 메모리 포화(7.9/8GB)로 162초/문항까지 느려져 30/200 에서 끊고 --resume 으로 이어 돈다(설정 동일)
cd "$(dirname "$0")/../../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
B=results/mh_only_20260928/baselines
flock .gpu.lock .venv/bin/python scripts/baselines/tablerag_yu.py --out-dir $B/tablerag_yu --stage answer --answer-ids $B/ids200.jsonl --resume \
  > $B/tablerag_yu/answer200_resume.log 2>&1
echo "[end] $(date -Is) yu_answer200_resume rc=$?"
