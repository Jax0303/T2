#!/usr/bin/env bash
# langchain_chunk 비교군: LangChain 공식 RAG 튜토리얼 파이프라인, MultiHiertt train (README.md 참고)
#   ./run.sh            검색 2,885 + 답변 882 (전체)
#   ./run.sh --limit 3  882 부분집합 앞 3건만 (연기 시험)
# GPU 는 .gpu.lock 으로 직렬화한다 — 인코더와 리더를 한 프로세스에 같이 올리지 않는다.
cd "$(dirname "$0")/../../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/mh_only_20260928/baselines/langchain_chunk
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD) args=$*"
t=$(date +%s)
flock .gpu.lock .venv/bin/python scripts/baselines/langchain_chunk.py --stage retrieve "$@" > $D/retrieve.log 2>&1
echo "[end] $(date -Is) retrieve rc=$? $(( $(date +%s) - t ))s"
t=$(date +%s)
flock .gpu.lock .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer "$@" > $D/answer.log 2>&1
echo "[end] $(date -Is) answer rc=$? $(( $(date +%s) - t ))s"
echo ALL_DONE
