#!/usr/bin/env bash
# run_all.sh(LangChain→RAPTOR→Chen)가 끝난 뒤 Yu 두 단계. 실행: setsid nohup bash results/mh_only_20260928/baselines/run_yu.sh > results/mh_only_20260928/baselines/run_yu.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
B=results/mh_only_20260928/baselines; P=.venv/bin/python
until grep -q ALL_DONE_FIRST4 $B/run_all.log 2>/dev/null; do sleep 60; done
step() { n=$1; log=$2; shift 2; t=$(date +%s); echo "[start] $(date -Is) $n"
  flock .gpu.lock "$@" > "$log" 2>&1; rc=$?
  echo "[end] $(date -Is) $n rc=$rc $(( $(date +%s) - t ))s"; [ "$rc" = 0 ] || { echo "STOPPED at $n"; exit 1; }; }
step yu_retrieve $B/tablerag_yu/retrieve.log $P scripts/baselines/tablerag_yu.py --out-dir $B/tablerag_yu --stage retrieve
step yu_answer   $B/tablerag_yu/answer.log   $P scripts/baselines/tablerag_yu.py --out-dir $B/tablerag_yu --stage answer
echo ALL_DONE
