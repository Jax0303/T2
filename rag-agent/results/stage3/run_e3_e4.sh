#!/bin/bash
# E3: in-domain CE reranker -> rerank p0 top-10 on clean single dev (hit@k dry-run, then reader rr10/rr10_3)
# E4: CoT answer mode on arithmetic (gold/top10) dev+test, single-cell 200-query control
cd /home/user/T2-1/rag-agent
COMMON="--embed-model models/bge-base-cell-ft-p1 --alpha 0.8 --title-mode page"
echo "=== E3 train ==="
PYTHONPATH=.:scripts .venv/bin/python scripts/train_cell_reranker.py --retriever models/bge-base-cell-ft-p1 --out models/ce-cell-p1
echo "EXIT $?"
echo "=== E3 dry-run hit@k (dev single clean) ==="
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON --split dev --population hitab_dev_lookup_all \
  --gold-file results/audit2/hitab_dev_lookup_all_gold.json --gold 0 --ks 1 3 10 --rerank-model models/ce-cell-p1 --rerank-ks 10 --dry-run --out /dev/null
echo "EXIT $?"
echo "=== E3 reader rr10/rr10_3 + top1/top3/top10 (dev single clean) ==="
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON --split dev --population hitab_dev_lookup_all \
  --gold-file results/audit2/hitab_dev_lookup_all_gold.json --gold 0 --ks 1 3 10 --rerank-model models/ce-cell-p1 --rerank-ks 10 --out results/stage3/e3_single_dev.jsonl
echo "EXIT $?"
echo "=== E4 cot arith dev ==="
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON --split dev --population hitab_dev_corpus_arith \
  --gold-file results/audit2/hitab_dev_corpus_arith_gold.json --gold 1 --ks 10 --answer-mode cot --out results/stage3/e4_arith_dev_cot.jsonl
echo "EXIT $?"
echo "=== E4 cot arith test ==="
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON --split test --population hitab_test_corpus_arith \
  --gold-file results/audit2/hitab_test_corpus_arith_gold.json --gold 1 --ks 10 --answer-mode cot --out results/stage3/e4_arith_test_cot.jsonl
echo "EXIT $?"
echo "=== E4 cot single dev control (200) ==="
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON --split dev --population hitab_dev_lookup_all \
  --gold-file results/audit2/hitab_dev_lookup_all_gold_first200.json --gold 1 --ks 10 --answer-mode cot --out results/stage3/e4_single_dev200_cot.jsonl
echo "EXIT $?"
echo "=== E4 value-mode arith dev/test on clean gold (paired control for cot) ==="
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON --split dev --population hitab_dev_corpus_arith \
  --gold-file results/audit2/hitab_dev_corpus_arith_gold.json --gold 1 --ks 10 --out results/stage3/e4_arith_dev_value.jsonl
echo "EXIT $?"
PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON --split test --population hitab_test_corpus_arith \
  --gold-file results/audit2/hitab_test_corpus_arith_gold.json --gold 1 --ks 10 --out results/stage3/e4_arith_test_value.jsonl
echo "EXIT $?"
echo E3E4_DONE
