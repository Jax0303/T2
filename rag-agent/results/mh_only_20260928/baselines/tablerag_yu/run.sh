#!/usr/bin/env bash
# TableRAG(Yu 2025) 원 방법 비교군. 사용: bash run.sh --stage retrieve|answer [--limit N] [--resume]
#   retrieve: 2,885 문항 1회차 텍스트 검색만(LLM·SQL 없음) → tablerag_yu_records.jsonl
#   answer  : 882 답변 집합 전체 파이프라인 → tablerag_yu_answer.jsonl
# GPU 잠금 .gpu.lock 아래에서 한 프로세스만 돈다.
cd "$(dirname "$0")/../../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928/baselines/tablerag_yu
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) vendor $(git -C vendor/tablerag_yu rev-parse --short HEAD) args: $*"
flock .gpu.lock .venv/bin/python scripts/baselines/tablerag_yu.py --out-dir $D "$@"
echo "[end] $(date -Is) rc=$?"
