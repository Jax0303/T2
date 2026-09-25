#!/usr/bin/env bash
# 2026-09-26 MultiHiertt c·e 재실행 (리더 없음). pre.py 통과 뒤 실행.
# 인자 = 기존 results/mh_arms/mh_cell_hv2{,_L1}.json 의 arguments. 임베딩 캐시는 빈 새 폴더 -> 새로 인코딩해 저장.
# --embed-overflow truncate = 기존 실행과 같은 처리(기존 코드는 검사 없이 모델 한도에서 잘랐다). 초과 수는 요약 JSON 에 남는다.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/recheck_20260926
for arm in "none mh_cell_hv2" "L1 mh_cell_hv2_L1"; do
  set -- $arm
  .venv/bin/python scripts/mh_arms.py --split train --unit cell --template s3c --row-text sentence \
    --header-rule v2 --label-rule "$1" --embed-model BAAI/bge-base-en-v1.5 --embed-overflow truncate \
    --alpha 0.7 --budget 20 --shard 50000 --dump-context 1 \
    --cache-dir .cache/recheck_20260926 --out-dir "$D" --tag "$2" > "$D/$2.log" 2>&1
done
.venv/bin/python "$D/analyze.py" > "$D/analyze.log" 2>&1
