#!/usr/bin/env bash
# 트리·기록(raptor_records.json)이 끝나면 원래 지시문 답변 단계(주 결과 아님)를 건너뛰고 LangChain 지시문 답변으로 간다.
set -uo pipefail
cd "$(dirname "$0")/../.."
S=results/main_lcprompt_20261012
until [ -f $S/raptor882/raptor_records.json ]; do sleep 20; done
pid=$(pgrep -f "scripts/baselines/raptor.py --limit 882" | while read p; do [ "$(ps -o comm= -p $p)" = python ] && echo $p; done | head -1)
echo "[skip] $(date -Is) raptor.py pid=$pid 원래 지시문 답변 단계 건너뜀"
[ -n "$pid" ] && kill $pid
while pgrep -f "results/main_lcprompt_20261012/run.sh" >/dev/null; do sleep 5; done
python3 - $S/raptor882/raptor_records.jsonl $S/raptor882_records.jsonl <<'PY'
import json, sys
with open(sys.argv[1]) as f, open(sys.argv[2], "x") as g:
    for l in f:
        r = json.loads(l)
        g.write(json.dumps({"query_id": r["query_id"], "kind": r["kind"], "layer": r["layer"], "m": len(r["gold_ids"]),
                            "question": r["question"], "gold_ids": r["gold_ids"], "context": r["context_2000"],
                            "correct": r["correct_2000"], "cells_in_context": r["cells_2000"]}, ensure_ascii=False) + "\n")
PY
echo "[start] $(date -Is) raptor882_lc"
flock /home/user/T2-1/rag-agent/.gpu.lock env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python \
  scripts/baselines/langchain_chunk.py --stage answer --records $S/raptor882_records.jsonl --out-dir $S/raptor882_lc --resume
echo "[end] $(date -Is) rc=$?"
