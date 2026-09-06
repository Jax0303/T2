#!/bin/bash
cd /home/user/T2-1/rag-agent
until grep -q "SCOPE_DONE\|Traceback" results/scope/run.log; do sleep 30; done
mkdir -p results/mh
echo "=== MH bi-encoder S3 ==="
PYTHONPATH=.:scripts .venv/bin/python scripts/mh_bi_encoder.py --scheme S3 --out results/mh/bi_S3.json
echo "EXIT $?"
echo MH_DONE
