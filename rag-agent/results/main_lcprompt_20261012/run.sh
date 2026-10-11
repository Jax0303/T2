#!/usr/bin/env bash
# PREREG-2026-10-12-main-baselines-lcprompt.md — RAPTOR 를 LangChain 지시문으로 (200, 이어서 882).
#   setsid nohup bash results/main_lcprompt_20261012/run.sh > results/main_lcprompt_20261012/run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
S=results/main_lcprompt_20261012; B=results/mh_only_20260928/baselines
RUN="flock /home/user/T2-1/rag-agent/.gpu.lock env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python"
conv() {  # RAPTOR 레코드 → langchain_chunk 답변 단계가 읽는 형식 (2,000토큰 입력)
  python3 - "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1]) as f, open(sys.argv[2], "x") as g:
    for l in f:
        r = json.loads(l)
        g.write(json.dumps({"query_id": r["query_id"], "kind": r["kind"], "layer": r["layer"], "m": len(r["gold_ids"]),
                            "question": r["question"], "gold_ids": r["gold_ids"], "context": r["context_2000"],
                            "correct": r["correct_2000"], "cells_in_context": len(r["cells_2000"])}, ensure_ascii=False) + "\n")
PY
}
echo "[start] $(date -Is) raptor200_lc"
[ -f $S/raptor200_records.jsonl ] || conv $B/raptor/raptor_records.jsonl $S/raptor200_records.jsonl
$RUN scripts/baselines/langchain_chunk.py --stage answer --limit 200 --records $S/raptor200_records.jsonl --out-dir $S/raptor200_lc --resume
echo "[start] $(date -Is) raptor882_build"
$RUN scripts/baselines/raptor.py --limit 882 --out-dir $S/raptor882 --tree-dir $B/raptor/trees --batch-size 128 --resume
echo "[start] $(date -Is) raptor882_lc"
[ -f $S/raptor882_records.jsonl ] || conv $S/raptor882/raptor_records.jsonl $S/raptor882_records.jsonl
$RUN scripts/baselines/langchain_chunk.py --stage answer --records $S/raptor882_records.jsonl --out-dir $S/raptor882_lc --resume
echo "[end] $(date -Is)"
