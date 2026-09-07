#!/bin/bash
# 학습 없는 arm(기성품 bge-base) 의 test 순위 덤프. 2026-09-07 사용자 결정: 인코더 학습 없음, test 만.
# α=0.8·page 는 results/baselines/BOARD.md 의 hybrid_cell 과 같은 조건 — @10 이 그 표와 일치해야 한다.
cd /home/ugh/T2/rag-agent
for pop in lookup_all lookup_multi corpus_arith; do
  echo "=== stock test $pop ==="
  PYTHONPATH=. python3 analysis/cell_rank_dump.py --dataset hitab --split test --population hitab_test_$pop \
    --embed-model BAAI/bge-base-en-v1.5 --alpha 0.8 --title-mode page \
    --gold-file results/audit2/hitab_test_${pop}_gold.json --out-dir results/stage1_stock
  echo "EXIT ${PIPESTATUS[0]}"
done
echo STOCK_DONE
