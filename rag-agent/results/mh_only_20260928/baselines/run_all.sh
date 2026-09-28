#!/usr/bin/env bash
# PREREG-2026-09-29-baselines-original.md 실행 순서 — GPU 1개, 순차. 각 단계는 flock .gpu.lock 안에서 앞에서 기다린다
# (개별 run.sh 는 쓰지 않는다: langchain 은 rc 를 $(date) 뒤에 읽고, chen 은 setsid 로 빠져 순서가 깨진다).
# 실행: cd rag-agent && setsid nohup bash results/mh_only_20260928/baselines/run_all.sh > results/mh_only_20260928/baselines/run_all.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
B=results/mh_only_20260928/baselines; P=.venv/bin/python
step() {  # $1 이름, $2 로그, 나머지 명령 — rc 는 명령 직후에 읽는다
  n=$1; log=$2; shift 2; t=$(date +%s); echo "[start] $(date -Is) $n"
  flock .gpu.lock "$@" > "$log" 2>&1; rc=$?
  echo "[end] $(date -Is) $n rc=$rc $(( $(date +%s) - t ))s"
  [ "$rc" = 0 ] || { echo "STOPPED at $n"; exit 1; }
}
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
step langchain_retrieve $B/langchain_chunk/retrieve.log $P scripts/baselines/langchain_chunk.py --stage retrieve
step langchain_answer   $B/langchain_chunk/answer.log   $P scripts/baselines/langchain_chunk.py --stage answer
step raptor             $B/raptor/full.log              $P scripts/baselines/raptor.py
PYTHONHASHSEED=0 step tablerag_chen $B/tablerag_chen/run.log $P scripts/baselines/tablerag_chen.py --out-dir $B/tablerag_chen
echo ALL_DONE_FIRST4  # Yu 두 단계는 run_yu.sh 로 따로
