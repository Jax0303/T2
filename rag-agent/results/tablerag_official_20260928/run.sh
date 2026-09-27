#!/usr/bin/env bash
# TableRAG 셀 검색 재구현 — 원본 숫자 열 판정(official)으로 검색만 6회 (PREREG-2026-09-28-tablerag-official-dtype.md).
# 인자는 --tablerag-dtype official 과 출력·캐시 경로만 results/rerun_20260926/run.sh 와 다르다.
#   cd rag-agent && setsid nohup bash results/tablerag_official_20260928/run.sh > results/tablerag_official_20260928/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"; export PYTHONPATH=.
R=results/tablerag_official_20260928; C=.cache/tablerag_official_20260928; CM=.cache/tablerag_official_20260928_mh
mkdir -p $R/hitab $R/mh $C $CM
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"

run() {  # run <out_dir> <tag> <script> <args...>
  local out=$1 tag=$2 script=$3; shift 3
  if [ -e "$out/$tag.json" ]; then echo "[skip] $tag"; return 0; fi
  echo "[run] $(date -Is) $tag"
  $PY scripts/$script "$@" --out-dir "$out" --tag "$tag" > "$out/$tag.log" 2>&1
  local rc=$?; echo "[end] $(date -Is) $tag rc=$rc"
}

HB="--split test --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --cache-dir $C"
for scope in gold split; do
  for m in leaf path; do
    run $R/hitab "hitab_test_${scope}_tablerag_${m}_official" retrieval_accuracy.py $HB --corpus $scope \
      --template s3c --unit tablerag --tablerag-colmode $m --tablerag-dtype official
  done
done
MB="--header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5"
for m in leaf path; do
  run $R/mh "mh_train_tablerag_${m}_official" mh_arms.py --split train $MB --cache-dir $CM \
    --unit tablerag --template s3c --tablerag-colmode $m --tablerag-dtype official
done
$PY $R/compare.py > $R/compare.log 2>&1; echo "[compare] rc=$?"
echo "[done] $(date -Is)"
