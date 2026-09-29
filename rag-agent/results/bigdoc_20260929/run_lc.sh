#!/usr/bin/env bash
# PREREG-2026-09-29-bigdoc.md 추가 등록: LangChain 청크(합친 문서). 표 전체 입력이 끝난 뒤 시작(GPU 공유 불가).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/bigdoc_20260929; mkdir -p $D/langchain_chunk
until grep -q ALL_DONE $D/run_full1.log 2>/dev/null; do sleep 30; done
echo "[start] $(date -Is) commit $(git rev-parse --short HEAD)"
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage langchain -- --stage retrieve > $D/langchain_chunk/retrieve.log 2>&1
echo "[end] $(date -Is) lc_retrieve rc=$? $(( $(date +%s) - t ))s"
t=$(date +%s)
.venv/bin/python scripts/mh_bigdoc.py --stage langchain -- --stage answer > $D/langchain_chunk/answer.log 2>&1
echo "[end] $(date -Is) lc_answer rc=$? $(( $(date +%s) - t ))s"
echo ALL_DONE
