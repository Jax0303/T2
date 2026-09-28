#!/usr/bin/env bash
# PREREG-2026-09-29-gold-only-answer.md. GPU 잠금을 잡은 채: 정답 셀만 입력 882 → Chen 200 이어 돌리기(--resume) → 멈춘 비교군 재개.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
O=results/gold_only_20260929
flock .gpu.lock bash -c '
echo "[start] $(date -Is) gold commit $(git rev-parse --short HEAD)"
.venv/bin/python scripts/answer_accuracy_mh.py --records results/mh_arms/mh_train_cell_hv3.3u_none_doc_records.jsonl \
  --scope doc --condition gold --split train --header-rule v3.3u --label-rule none --prompt cot --max-tokens 384 \
  --batch-size 128 --same-queries-as results/mh_arms/cap300_20260924/cell_uniq.jsonl \
  --out '$O'/mh_gold_cot384.jsonl > '$O'/mh_gold_cot384.log 2>&1
echo "[end] $(date -Is) gold rc=$?"
PYTHONHASHSEED=0 .venv/bin/python scripts/baselines/tablerag_chen.py --out-dir results/mh_only_20260928/baselines/tablerag_chen --limit 200 --resume \
  > results/mh_only_20260928/baselines/tablerag_chen/run200_resume.log 2>&1
echo "[end] $(date -Is) chen_resume rc=$?"
'
kill -CONT 1170410 1171368
echo "[resumed] $(date -Is) run_rest.sh, raptor flock"
echo ALL_DONE
