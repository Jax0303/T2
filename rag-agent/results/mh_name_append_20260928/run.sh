#!/usr/bin/env bash
# PREREG-2026-09-28-mh-name-append.md: 방법 3개 × (원래 질문, 정답 열 이름 붙인 질문), 정답 셀 1개 283문항
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_name_append_20260928
MB="--split train --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,20 --only-uids $D/uids.json"
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
arm() {
  case $1 in
    s3c)   echo "--cache-dir .cache/rerun_20260926_mh --unit cell --template s3c" ;;
    chunk) echo "--cache-dir .cache/rerun_20260926_mh --unit chunk --template s3c --embed-overflow truncate" ;;
    leaf)  echo "--cache-dir .cache/tablerag_official_20260928_mh --unit tablerag --template s3c --tablerag-colmode leaf --tablerag-dtype official" ;;
  esac
}
for a in s3c chunk leaf; do
  for cond in base append; do
    extra=""; [ $cond = append ] && extra="--question-suffix $D/suffix.json"
    .venv/bin/python scripts/mh_arms.py $MB $(arm $a) $extra --out-dir $D --tag ${a}_${cond} > $D/${a}_${cond}.log 2>&1
    echo "[end] $(date -Is) ${a}_${cond} rc=$?"
  done
done
.venv/bin/python $D/analyze.py > $D/analyze.log 2>&1; echo "[end] $(date -Is) analyze rc=$?"
echo ALL_DONE
