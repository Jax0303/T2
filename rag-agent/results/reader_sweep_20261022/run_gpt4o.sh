#!/usr/bin/env bash
# 리더 교체 실험 외부 리더: GPT-4o × P* — D 1회 + O1 2회(반복 재현), 882, 출력 4,096, temperature 0, seed 없음 — PREREG-2026-10-22-reader-sweep.md
#   setsid nohup bash results/reader_sweep_20261022/run_gpt4o.sh > results/reader_sweep_20261022/run_gpt4o.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/reader_sweep_20261022; V=R1X1F1B2D1C1E0
D=results/format_dev_20261017/D882_records.jsonl; O1=results/ceiling_20261019/O1_records.jsonl
run() {  # 이름 arm 레코드
  echo "[start $1] $(date -Is)"
  env PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 .venv/bin/python scripts/prompt_sweep.py --engine openai --model gpt-4o --key-env OPENAI_API_KEY_OLD --workers 6 \
    --records "$3" --arm "$2" --variants "$V" --out-dir "$O/$1" --max-new-tokens 4096 --resume
  echo "[exit $? $1] $(date -Is)"
  env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_formats2.py "$O/$1" --gold train | tee "$O/score_$1.txt"
}
run gpt4o_D_run1 D "$D"
run gpt4o_O1_run1 O1 "$O1"
run gpt4o_O1_run2 O1 "$O1"
echo "[all done] $(date -Is)"
