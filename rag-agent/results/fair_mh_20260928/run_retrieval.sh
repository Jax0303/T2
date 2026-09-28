#!/usr/bin/env bash
# 2026-09-28 공정 비교(PREREG-2026-09-28-fair-mh.md): 표 5-5 설정 그대로, 순위 단위·셀 좌표를 기록에 추가해 재검색 (캐시 재사용)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/fair_mh_20260928/retrieval
MB="--split train --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20 --cache-dir .cache/rerun_20260926_mh"
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
arm() { case $1 in
  s3c)         echo "--unit cell --template s3c" ;;
  chunk)       echo "--unit chunk --template s3c --embed-overflow truncate" ;;
  trag_hetero) echo "--unit trag_hetero --template s3c --embed-overflow truncate" ;;
  table)       echo "--unit table --template s3c --embed-overflow truncate" ;;
  row)         echo "--unit row --template s3c --row-text values --embed-overflow truncate" ;;
esac; }
for a in ${ARMS:-s3c chunk trag_hetero}; do
  .venv/bin/python scripts/mh_arms.py $MB $(arm $a) --out-dir $D --tag mh_train_$a > $D/mh_train_$a.log 2>&1
  echo "[end] $(date -Is) $a rc=$?"
done
echo ALL_DONE
