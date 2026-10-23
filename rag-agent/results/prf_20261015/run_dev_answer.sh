#!/usr/bin/env bash
# PREREG-2026-10-15-prf.md dev 답변만 다시(검색·전달 레코드는 run_dev.sh 결과 그대로). 첫 실행은 --split validation 누락으로 멈춤(이탈 기록).
#   setsid nohup bash results/prf_20261015/run_dev_answer.sh > results/prf_20261015/run_dev_answer.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/prf_20261015; R=$D/dev
IDS=results/reader_format_20260927/mh_dev_pop332.jsonl
for c in prf1_path prf3_path prf5_path prf1_full prf3_full prf5_full; do
  echo "[answer] $(date -Is) $c"
  rm -rf $R/ans_$c
  .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer --split validation --prompt original --resume --records $R/${c}_input.jsonl --ids $IDS --out-dir $R/ans_$c > $R/ans_$c.log 2>&1 || echo "answer failed $c"
  echo "[done] $(date -Is) $c $(wc -l < $R/ans_$c/langchain_chunk_answer.jsonl 2>/dev/null)"
done
echo "[all done] $(date -Is)"
