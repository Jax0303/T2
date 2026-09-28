#!/usr/bin/env bash
# PREREG-2026-09-28-mh-only.md GATE 1: 순위 목록(상위 20 단위 셀 좌표·토큰 수)이 필요한 arm 검색 1회 (리더 없음, 임베딩 캐시 재사용)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928/retrieval; mkdir -p $D
MB="--split train --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20"
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
arm() { case $1 in
  s3c)           echo "--cache-dir .cache/rerun_20260926_mh --unit cell --template s3c" ;;
  chunk)         echo "--cache-dir .cache/rerun_20260926_mh --unit chunk --template s3c --embed-overflow truncate" ;;
  trag_hetero)   echo "--cache-dir .cache/rerun_20260926_mh --unit trag_hetero --template s3c --embed-overflow truncate" ;;
  table)         echo "--cache-dir .cache/rerun_20260926_mh --unit table --template s3c --embed-overflow truncate" ;;
  rowcol)        echo "--cache-dir .cache/rerun_20260926_mh --unit rowcol --template s3c --row-text values --embed-overflow truncate" ;;
  tablerag_leaf) echo "--cache-dir .cache/tablerag_official_20260928_mh --unit tablerag --template s3c --tablerag-colmode leaf --tablerag-dtype official" ;;
  tablerag_path) echo "--cache-dir .cache/tablerag_official_20260928_mh --unit tablerag --template s3c --tablerag-colmode path --tablerag-dtype official" ;;
esac; }
for a in s3c chunk trag_hetero table rowcol tablerag_leaf tablerag_path; do
  t=$(date +%s)
  .venv/bin/python scripts/mh_arms.py $MB $(arm $a) --out-dir $D --tag mh_train_$a > $D/mh_train_$a.log 2>&1
  echo "[end] $(date -Is) $a rc=$? $(( $(date +%s) - t ))s"
done
# dev 332 (validation, 표 근거만 필요한 문항), 본 방법, 예산 50·순위 50 — §3.5 곡선·§6 필터용 (규칙 선택 전용)
t=$(date +%s)
.venv/bin/python scripts/mh_arms.py --split validation --header-rule v3.3u --alpha 0.7 --budget 50 --dump-ranked 50 \
  --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,3,5,10,15,20,30,50 --cache-dir .cache/rerun_20260926_mh \
  --unit cell --template s3c --out-dir $D --tag mh_dev_s3c_b50 > $D/mh_dev_s3c_b50.log 2>&1
echo "[end] $(date -Is) dev_s3c_b50 rc=$? $(( $(date +%s) - t ))s"
echo ALL_DONE
