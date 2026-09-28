#!/usr/bin/env bash
# RAPTOR 비교군 (scripts/baselines/raptor.py). GPU 잠금(flock) 안에서 한 프로세스만 돈다.
#   bash results/mh_only_20260928/baselines/raptor/run.sh --limit 3 --out-dir results/mh_only_20260928/baselines/raptor/smoke3
#   setsid nohup bash results/mh_only_20260928/baselines/raptor/run.sh > results/mh_only_20260928/baselines/raptor/full.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../../.."
exec flock /home/user/T2-1/rag-agent/.gpu.lock env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    .venv/bin/python scripts/baselines/raptor.py "$@"
