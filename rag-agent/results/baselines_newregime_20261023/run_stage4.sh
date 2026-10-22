#!/usr/bin/env bash
# 4단계(vLLM 서버 경로, 사용자 지시 10-23 01:06 "빠르게 효율적으로"): 시험 3문항(4 비교군 동시) → 전체 병렬(Chen 3조각·Yu 2조각·RAPTOR·LangChain)
#   서버: vllm serve Qwen/Qwen3.5-4B --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.89 --max-num-seqs 32 --enable-prefix-caching --language-model-only --seed 42
#   사용: bash run_stage4.sh pilot | full
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/baselines_newregime_20261023; R="vllm:Qwen/Qwen3.5-4B"; IDS=results/mh_arms/cap300_20260924/cell_uniq.jsonl
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_BASE_URL=http://localhost:8000/v1
P=.venv/bin/python
run() { n=$1; shift; t=$(date +%s); echo "[start $n] $(date -Is)"; "$@" > "$O/$n.log" 2>&1; echo "[end $n rc=$? $(( $(date +%s)-t ))s] $(date -Is)"; }
case "${1:-pilot}" in
  pilot)
    run v_pilot_chen env PYTHONHASHSEED=0 $P scripts/baselines/tablerag_chen.py --out-dir $O/vpilot/chen --limit 3 --max-tokens 512 --reader "$R" --resume &
    run v_pilot_yu $P scripts/baselines/tablerag_yu.py --out-dir $O/vpilot/yu --stage answer --answer-ids $IDS --limit 3 --reader "$R" --embed-device cpu --reranker-device cpu --resume &
    run v_pilot_raptor $P scripts/baselines/raptor.py --out-dir $O/vpilot/raptor --limit 3 --reader "$R" --batch-size 1 --embed-device cpu --resume &
    run v_pilot_langchain $P scripts/baselines/langchain_chunk.py --stage answer --out-dir $O/vpilot/langchain --records results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl --reader "$R" --batch-size 1 --limit 3 --resume &
    wait; echo "[pilot done] $(date -Is)" ;;
  full)
    run langchain $P scripts/baselines/langchain_chunk.py --stage answer --out-dir $O/langchain_chunk_vllm --records results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl --reader "$R" --batch-size 1 --resume &
    run raptor $P scripts/baselines/raptor.py --out-dir $O/raptor --limit 882 --reader "$R" --batch-size 1 --embed-device cpu --resume &
    for k in 0 1; do run tablerag_yu_s$k $P scripts/baselines/tablerag_yu.py --out-dir $O/tablerag_yu_s$k --stage answer --answer-ids $IDS --shard $k/2 --reader "$R" --embed-device cpu --reranker-device cpu --resume & done
    for k in 0 1 2; do run tablerag_chen_s$k env PYTHONHASHSEED=0 $P scripts/baselines/tablerag_chen.py --out-dir $O/tablerag_chen_s$k --limit 882 --shard $k/3 --max-tokens 512 --reader "$R" --resume & done
    wait; echo "[full done] $(date -Is)" ;;
esac
