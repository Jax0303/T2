#!/bin/bash
# 사전등록 PREREG-2026-09-09-encoder-robustness.md — 조작 변인은 --embed-model 하나.
# arm 인자는 각 *.json 에서 복원했고, n_units 가 지문 역할을 한다(사전등록 §6).
set -u
cd /home/user/T2-1/rag-agent
E="BAAI/bge-large-en-v1.5"
run () {  # run <tag> <args...>
  local tag=$1; shift
  echo "=== $tag ==="
  PYTHONPATH=. .venv/bin/python scripts/retrieval_accuracy.py \
    --corpus split --alpha 0.7 --budget 20 --dump-context 20 \
    --embed-model "$E" --tag "${tag}_lg" "$@"
  echo "EXIT $? $tag"
}
run t_s3c_hybrid      --unit cell   --template s3c
run t_mt2net_hybrid   --unit cell   --template mt2net
run t_row_hybrid      --unit row    --row-text sentence
run t_row_values      --unit row    --row-text values
run t_rowcol_values   --unit rowcol --row-text values
run t_trag_hetero     --unit trag_hetero --chunk-chars 1000
run t_trag_hetero_tok --unit trag_hetero --chunk-chars 2400
run t_tablerag_leaf   --unit tablerag --tablerag-colmode leaf
run t_tablerag_path   --unit tablerag --tablerag-colmode path
run t_chunk1000       --unit chunk  --chunk-chars 1000
echo ENCODER_SWEEP_DONE
