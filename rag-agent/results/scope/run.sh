#!/bin/bash
cd /home/user/T2-1/rag-agent
echo "=== train p2 ==="
PYTHONPATH=. .venv/bin/python scripts/finetune_from_pairs.py --pairs results/scope/train_pairs.jsonl --out models/bge-base-scope-p2
echo "EXIT $?"
for m in p1 p2; do
  [ $m = p1 ] && M=models/bge-base-cell-ft-p1 || M=models/bge-base-scope-p2
  for sp in dev test; do for pop in corpus_arith lookup_multi lookup_all; do
    echo "=== bench $m $sp $pop ==="
    PYTHONPATH=.:scripts:analysis .venv/bin/python analysis/scope_bench.py bench --split $sp --population hitab_${sp}_${pop} \
      --gold-file results/audit2/hitab_${sp}_${pop}_gold.json --embed-model $M --out results/scope/${m}_${sp}_${pop}.json
    echo "EXIT $?"
  done; done
done
echo SCOPE_DONE
