#!/usr/bin/env bash
# 2단계: 1단계 종료 대기 → 3문항 시험(Chen·Yu·RAPTOR, 실패 시 중단) → 전체(RAPTOR 882 → Yu 882 → Chen 882) — PREREG-2026-10-23-baselines-newregime.md
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/baselines_newregime_20261023; R="local:Qwen/Qwen3.5-4B?quantization=none"; IDS=results/mh_arms/cap300_20260924/cell_uniq.jsonl
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
P=.venv/bin/python
while ! grep -q "^\[stage1 done\]" $O/run_stage1.log 2>/dev/null; do sleep 60; done
step() { n=$1; shift; t=$(date +%s); echo "[start $n] $(date -Is)"; "$@" > "$O/$n.log" 2>&1; rc=$?; echo "[end $n rc=$rc $(( $(date +%s)-t ))s] $(date -Is)"; return $rc; }
step pilot_chen env PYTHONHASHSEED=0 $P scripts/baselines/tablerag_chen.py --out-dir $O/pilot/chen --limit 3 --max-tokens 512 --reader "$R" --resume || { echo "[stop] pilot_chen failed"; exit 1; }
step pilot_yu $P scripts/baselines/tablerag_yu.py --out-dir $O/pilot/yu --stage answer --answer-ids $IDS --limit 3 --reader "$R" --embed-device cpu --reranker-device cpu --resume || { echo "[stop] pilot_yu failed"; exit 1; }
step pilot_raptor $P scripts/baselines/raptor.py --out-dir $O/pilot/raptor --limit 3 --reader "$R" --batch-size 1 --resume || { echo "[stop] pilot_raptor failed"; exit 1; }
step raptor $P scripts/baselines/raptor.py --out-dir $O/raptor --limit 882 --reader "$R" --batch-size 1 --resume
step tablerag_yu $P scripts/baselines/tablerag_yu.py --out-dir $O/tablerag_yu --stage answer --answer-ids $IDS --reader "$R" --embed-device cpu --reranker-device cpu --resume
step tablerag_chen env PYTHONHASHSEED=0 $P scripts/baselines/tablerag_chen.py --out-dir $O/tablerag_chen --limit 882 --max-tokens 512 --reader "$R" --resume
echo "[stage2 done] $(date -Is)"
