#!/usr/bin/env bash
# PREREG-2026-09-28-mh-only.md GATE 1 보완: 셀 단위 arm 4개 순위 100 재검색 (리더 없음, 임베딩 캐시 재사용). 인자는 run_retrieval.sh 와 같고 --dump-ranked 100 만 다름.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
until grep -q "k=50 rc" results/mh_only_20260928/dev_k_answer.log; do sleep 30; done   # dev k=50 답변과 GPU 공유 불가
D=results/mh_only_20260928/retrieval_r100; mkdir -p $D
MB="--split train --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20 --dump-ranked 100"
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
arm() { case $1 in
  s3c)           echo "--cache-dir .cache/rerun_20260926_mh --unit cell --template s3c" ;;
  rowcol)        echo "--cache-dir .cache/rerun_20260926_mh --unit rowcol --template s3c --row-text values --embed-overflow truncate" ;;
  tablerag_leaf) echo "--cache-dir .cache/tablerag_official_20260928_mh --unit tablerag --template s3c --tablerag-colmode leaf --tablerag-dtype official" ;;
  tablerag_path) echo "--cache-dir .cache/tablerag_official_20260928_mh --unit tablerag --template s3c --tablerag-colmode path --tablerag-dtype official" ;;
esac; }
for a in s3c tablerag_leaf tablerag_path rowcol; do
  t=$(date +%s)
  .venv/bin/python scripts/mh_arms.py $MB $(arm $a) --out-dir $D --tag mh_train_$a > $D/mh_train_$a.log 2>&1
  echo "[end] $(date -Is) $a rc=$? $(( $(date +%s) - t ))s"
done
echo ALL_DONE
