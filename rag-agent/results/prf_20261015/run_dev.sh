#!/usr/bin/env bash
# PREREG-2026-10-15-prf.md dev 332: 후보 6 (N 1/3/5 × full/path) + prf0 재현 확인. 검색 → 전달 레코드 → Qwen 답변.
#   setsid nohup bash results/prf_20261015/run_dev.sh > results/prf_20261015/run_dev.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/prf_20261015; R=$D/dev
MB="--split validation --header-rule v3.3u --alpha 0.7 --budget 100 --dump-ranked 100 --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20,50,75,100 --cache-dir .cache/mh_val_20261015 --unit cell --template s3c --only-uids $D/dev_ids332.json"
LC=results/cellform_dev_20261012/lc_retrieve/langchain_chunk_records.jsonl
IDS=results/reader_format_20260927/mh_dev_pop332.jsonl
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
for c in prf0 prf1_path prf3_path prf5_path prf1_full prf3_full prf5_full; do
  n=${c:3:1}; t=${c#*_}; [ "$c" = prf0 ] && t=full
  echo "[retrieve] $(date -Is) $c"
  .venv/bin/python scripts/mh_arms.py $MB --prf $n --prf-text $t --out-dir $R --tag dev_$c > $R/dev_$c.log 2>&1 || { echo "retrieve failed $c"; continue; }
  .venv/bin/python $D/build_inputs.py validation $R/dev_${c}_records.jsonl $D/dev_ids332.json $LC $R/${c}_input.jsonl $( [ "$c" = prf0 ] && echo "--check results/cellform_dev_20261012/C_records.jsonl" ) 2>&1 | grep -v Warning
  [ "$c" = prf0 ] && continue
  echo "[answer] $(date -Is) $c"
  .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer --prompt original --resume --records $R/${c}_input.jsonl --ids $IDS --out-dir $R/ans_$c > $R/ans_$c.log 2>&1 || echo "answer failed $c"
  echo "[done] $(date -Is) $c $(wc -l < $R/ans_$c/langchain_chunk_answer.jsonl 2>/dev/null)"
done
echo "[all done] $(date -Is)"
