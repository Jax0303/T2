#!/usr/bin/env bash
# PREREG-2026-09-29-chen-max-tokens.md — TableRAG(Chen) 200 문항, 호출당 생성 상한 512 (원본 128). 나머지 설정은 tablerag_chen/ 실행과 같다.
cd "$(dirname "$0")/../../../.." || exit 1
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
D=results/mh_only_20260928/baselines/tablerag_chen_mt512
echo "[start-wait] $(date -Is) commit $(git rev-parse --short HEAD)"
flock .gpu.lock .venv/bin/python scripts/baselines/tablerag_chen.py --out-dir $D --limit 200 --max-tokens 512 > $D/run200.log 2>&1
echo "[end] $(date -Is) rc=$?"
