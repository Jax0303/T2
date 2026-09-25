#!/usr/bin/env bash
# 2026-09-26 라벨 벡터 섞기 (리더 없음): 셀 벡터 = normalize(α·c 벡터 + (1−α)·표 라벨 벡터), α ∈ {1.0 … 0.5}.
# 셀 문장·BM25 는 c 그대로, 검색 가중 .7·20셀.
# HiTab: c = s2 (results/retrieval_accuracy/t_s2_{gold,split}_labelabl 와 같은 인자), 라벨 = s3c 문장에 들어가는 제목.
#        gold = 질문의 표 안, split = test 표 538개를 한 색인에.
# MultiHiertt: c = 머리글 v2·라벨 없음 (results/recheck_20260926/mh_cell_hv2 와 같은 인자, 그 임베딩 캐시를 읽는다), 라벨 = L1.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/labelmix_20260926
for m in 1.0 0.9 0.8 0.7 0.6 0.5; do
  for c in gold split; do
    .venv/bin/python scripts/retrieval_accuracy.py --split test --data-dir data/hitab --corpus "$c" --template s2 \
      --unit cell --row-text sentence --embed-model BAAI/bge-base-en-v1.5 --alpha 0.7 --budget 20 --label-mix "$m" \
      --cache-dir .cache/retrieval_accuracy --dump-context 1 --out-dir "$D" --tag "t_s2_${c}_mix$m" > "$D/t_s2_${c}_mix$m.log" 2>&1
  done
done
for m in 1.0 0.9 0.8 0.7 0.6 0.5; do
  .venv/bin/python scripts/mh_arms.py --split train --unit cell --template s3c --row-text sentence \
    --header-rule v2 --label-rule none --label-mix "$m" --embed-model BAAI/bge-base-en-v1.5 --embed-overflow truncate \
    --alpha 0.7 --budget 20 --shard 50000 --dump-context 1 \
    --cache-dir .cache/recheck_20260926 --out-dir "$D" --tag "mh_cell_hv2_mix$m" > "$D/mh_cell_hv2_mix$m.log" 2>&1
done
.venv/bin/python "$D/analyze.py" > "$D/analyze.log" 2>&1
