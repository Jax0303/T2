#!/usr/bin/env bash
# tablerag_chen — Google TableRAG(Chen 2024) 공식 코드 전체 파이프라인, MultiHiertt 같은 조건.
#   bash results/mh_only_20260928/baselines/tablerag_chen/run.sh              # 전체: 검색 2,885 + 답변 882 (여기에 씀)
#   bash results/mh_only_20260928/baselines/tablerag_chen/run.sh --limit 3    # 점검: 882 부분집합 앞 3건 → smoke3/ 에 씀
#   bash results/mh_only_20260928/baselines/tablerag_chen/run.sh --resume     # 끊긴 전체 실행 이어 쓰기
# GPU 1개 공유: flock .gpu.lock 로 LLM 프로세스 하나만. 진행은 $D/run.log 와 *.jsonl 줄 수로 본다.
set -euo pipefail
cd "$(dirname "$0")/../../../.."                     # rag-agent/
D=results/mh_only_20260928/baselines/tablerag_chen
LIMIT=""; for ((i=1; i<=$#; i++)); do [[ "${!i}" == "--limit" ]] && j=$((i+1)) && LIMIT="${!j}"; done
[[ -n "$LIMIT" ]] && D="$D/smoke$LIMIT"
mkdir -p "$D"
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONHASHSEED=0                              # 원본이 검색 결과를 set() 으로 합친다 — 순서를 고정
touch .gpu.lock
setsid nohup flock .gpu.lock .venv/bin/python scripts/baselines/tablerag_chen.py --out-dir "$D" "$@" \
    > "$D/run.log" 2>&1 < /dev/null &
echo "started pid $! -> $D/run.log"
