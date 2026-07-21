#!/usr/bin/env bash
# Run the agent-pipeline ablation suite (run_eval.py) end to end.
#
# Usage: bash scripts/run_all_experiments.sh <HITAB_DIR> <CHROMA_DIR> [extra run_eval args...]
# Example:
#   bash scripts/run_all_experiments.sh data/hitab data/chroma_db \
#        --llm groq:llama-3.3-70b-versatile --retriever-device cpu
#
# Paths are resolved relative to the rag-agent package root, so this works from
# anywhere. Anything after the two directories is forwarded to every run.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <HITAB_DIR> <CHROMA_DIR> [extra run_eval args...]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$1"; shift
CHROMA_DIR="$1"; shift

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"

CONFIGS=(
    "v3.1_baseline           configs/v3.1_baseline.yaml"
    "decomposition           configs/decomposition.yaml"
    "ablation_no_symbolic    configs/ablation_no_symbolic.yaml"
    "ablation_oracle         configs/ablation_oracle_retrieval.yaml"
    "decomposition_oracle    configs/decomposition_oracle.yaml"
)

i=0
for entry in "${CONFIGS[@]}"; do
    i=$((i + 1))
    read -r name cfg <<<"$entry"
    echo "=== [$i/${#CONFIGS[@]}] $name ($cfg) ==="
    "$PY" scripts/run_eval.py \
        --config "$cfg" \
        --data-dir "$DATA_DIR" \
        --chroma-dir "$CHROMA_DIR" \
        "$@"
done

echo ""
echo "Done. Per-run JSON (config, per-class metrics, per-query rows) is under results/."
echo "Headline answer metric is hmtEM (HiTab's own scorer); NM is a lenient diagnostic."
