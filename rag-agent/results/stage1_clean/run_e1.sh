#!/bin/bash
cd /home/user/T2-1/rag-agent
echo "=== train p1 ==="
PYTHONPATH=.:scripts .venv/bin/python scripts/finetune_cell_encoder.py --base BAAI/bge-base-en-v1.5 --out models/bge-base-cell-ft-p1 \
  --population hitab_train_fit_alltypes --multi-gold all --title-mode page --neg-per-query 4 --neg-cross 0 --epochs 2 --batch-size 32 --lr 2e-5
echo "EXIT $?"
for sp in dev test; do for pop in lookup_all lookup_multi corpus_arith; do
  echo "=== rank p1 $sp $pop ==="
  PYTHONPATH=. .venv/bin/python analysis/cell_rank_dump.py --dataset hitab --split $sp --population hitab_${sp}_${pop} \
    --embed-model models/bge-base-cell-ft-p1 --alpha 0.8 --title-mode page --gold-file results/audit2/hitab_${sp}_${pop}_gold.json --out-dir results/stage1_clean/p1
  echo "EXIT $?"
done; done
echo E1_DONE
