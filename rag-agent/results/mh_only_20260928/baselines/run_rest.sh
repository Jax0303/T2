#!/usr/bin/env bash
# 2026-09-29 06:1x 시간 제약: Chen·Yu 답변은 882 중 앞 200 문항(ids200.jsonl). 실행: setsid nohup bash results/mh_only_20260928/baselines/run_rest.sh > results/mh_only_20260928/baselines/run_rest.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
B=results/mh_only_20260928/baselines; P=.venv/bin/python
step() { n=$1; log=$2; shift 2; t=$(date +%s); echo "[start] $(date -Is) $n"
  flock .gpu.lock "$@" > "$log" 2>&1; rc=$?
  echo "[end] $(date -Is) $n rc=$rc $(( $(date +%s) - t ))s"; [ "$rc" = 0 ] || echo "FAILED $n (continuing)"; }
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
PYTHONHASHSEED=0 step tablerag_chen200 $B/tablerag_chen/run200.log $P scripts/baselines/tablerag_chen.py --out-dir $B/tablerag_chen --limit 200
step yu_retrieve $B/tablerag_yu/retrieve.log $P scripts/baselines/tablerag_yu.py --out-dir $B/tablerag_yu --stage retrieve
step yu_answer200 $B/tablerag_yu/answer200.log $P scripts/baselines/tablerag_yu.py --out-dir $B/tablerag_yu --stage answer --answer-ids $B/ids200.jsonl
echo ALL_DONE
