#!/bin/bash
# 리더 사다리 (PREREG-2026-09-05-reader-ladder.md). 순서: 다중 dev/test -> 산술 dev/test -> 단일 test.
cd /home/user/T2-1/rag-agent
COMMON="--embed-model models/bge-base-cell-ft-p0 --alpha 0.8 --title-mode page --gold 1"
run() { echo "=== $* ==="; PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py $COMMON "$@"; echo "EXIT ${PIPESTATUS[0]} $?"; }
run --split dev  --population hitab_dev_lookup_multi   --ks 1 3 10 --oracle-ks 10 --out results/stage3/multi_dev.jsonl
run --split test --population hitab_test_lookup_multi  --ks 1 3 10 --oracle-ks 10 --out results/stage3/multi_test.jsonl
run --split dev  --population hitab_dev_corpus_arith   --ks 1 3 10 20 --oracle-ks 10 20 --out results/stage3/arith_dev.jsonl
run --split test --population hitab_test_corpus_arith  --ks 1 3 10 20 --oracle-ks 10 20 --out results/stage3/arith_test.jsonl
run --split test --population hitab_test_lookup_all    --ks 1 3 10 --oracle-ks 10 --out results/stage3/single_test.jsonl
echo LADDER_DONE
