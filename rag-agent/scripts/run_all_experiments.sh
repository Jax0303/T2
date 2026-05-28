#!/usr/bin/env bash
# Run the full experiment suite for the thesis.
# Usage: bash scripts/run_all_experiments.sh --data-dir /path/to/HiTab --chroma-dir /path/to/chroma
set -euo pipefail

DATA_DIR="${1:?Usage: $0 --data-dir DIR --chroma-dir DIR}"
shift
CHROMA_DIR="${1:?}"
shift

SCRIPT="python rag-agent/scripts/run_eval.py"
COMMON="--data-dir $DATA_DIR --chroma-dir $CHROMA_DIR"

echo "=== [1/6] v3.1 baseline (original extractor) ==="
$SCRIPT $COMMON --config rag-agent/configs/v3.1_baseline.yaml

echo "=== [2/6] Decomposition extractor ==="
$SCRIPT $COMMON --config rag-agent/configs/decomposition.yaml

echo "=== [3/6] Ablation: no verifier ==="
$SCRIPT $COMMON --config rag-agent/configs/ablation_no_verify.yaml

echo "=== [4/6] Ablation: no symbolic ==="
$SCRIPT $COMMON --config rag-agent/configs/ablation_no_symbolic.yaml

echo "=== [5/6] Ablation: oracle retrieval ==="
$SCRIPT $COMMON --config rag-agent/configs/ablation_oracle_retrieval.yaml

echo "=== [6/6] Decomposition + oracle retrieval ==="
$SCRIPT $COMMON --config rag-agent/configs/decomposition_oracle.yaml

echo ""
echo "All experiments complete. Run 'python rag-agent/scripts/aggregate_runs.py rag-agent/results/*.json' to compare."
