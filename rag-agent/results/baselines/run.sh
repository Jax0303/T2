#!/bin/bash
cd /home/user/T2-1/rag-agent
for sp in dev test; do for pop in lookup_all lookup_multi corpus_arith; do
  echo "=== $sp $pop ==="
  PYTHONPATH=.:scripts:analysis .venv/bin/python analysis/baseline_board.py --split $sp --population hitab_${sp}_${pop} --gold-file results/audit2/hitab_${sp}_${pop}_gold.json
  echo "EXIT $?"
done; done
echo BASELINES_DONE
