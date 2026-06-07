#!/usr/bin/env bash
# Phase 3 라우팅 ablations — Groq 재실행 (빠름), hard-class dev per_class=50.
# 4개 ablation을 일관된 동일 LLM으로 순차 실행 → A4 oracle-router는 사후 합성.
set -uo pipefail
cd /home/user/T2-1/rag-agent
PY=/home/user/T2/hart-table-retrieval/.venv/bin/python
export LLM_BACKEND=groq
export GROQ_MODEL=llama-3.1-8b-instant
PC=50

run () {  # $1=ablation $2=outfile
  local abl="$1" out="$2"
  if [ -f "results/$out" ]; then echo "[skip] $out exists"; return; fi
  echo "=== $(date +%H:%M) ablation=$abl → $out (groq=$GROQ_MODEL pc=$PC) ==="
  "$PY" scripts/codegen_eval.py --ablation "$abl" --per-class "$PC" --quiet --out "$out" \
    2>&1 | grep -aviE 'Batches|it/s|s/it|Loading weights|FutureWarning|torch._check'
}

run adaptive        phase3_FULL_groq.json
run always-codegen  phase3_A1_vector_groq.json
run always-original phase3_A2_structured_groq.json
run always-keyword  phase3_A3_nostruct_groq.json
echo "=== ALL DONE $(date +%H:%M) ==="
