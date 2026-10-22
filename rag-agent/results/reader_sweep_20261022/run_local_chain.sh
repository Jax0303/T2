#!/usr/bin/env bash
# 로컬 사슬: Qwen3-4B-Instruct-2507 bf16 × P* × {D, O1}(882, 4,096, 1회) → Qwen3.5-4B O1 앞 50문항 HF 한 건씩 greedy 2회(재현 점검) — PREREG-2026-10-22-reader-sweep.md
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/reader_sweep_20261022; M=Qwen/Qwen3-4B-Instruct-2507; V=R1X1F1B2D1C1E0
D=results/format_dev_20261017/D882_records.jsonl; O1=results/ceiling_20261019/O1_records.jsonl
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
MML=$(env PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 .venv/bin/python - "$M" "$V" "$D" "$O1" <<'PY'
import json, sys
from transformers import AutoTokenizer
import prompt_variants as pv
M, V, *files = sys.argv[1:]
tok = AutoTokenizer.from_pretrained(M); m = 0
for f in files:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        p = tok.apply_chat_template([{"role": "user", "content": pv.build(V, r["question"], r["context"])}], tokenize=False, add_generation_prompt=True)
        m = max(m, len(tok(p).input_ids))
print("max prompt tokens", m, file=sys.stderr)
print(8192 if m + 4200 <= 8192 else (10240 if m + 4200 <= 10240 else 12288))
PY
)
echo "[start 2507 seqs=24 max-model-len=$MML] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len $MML --max-num-seqs 24 \
  --records $D,$O1 --arm D,O1 --variants "$V" --out-dir $O/run_qwen3_4b_2507 --max-new-tokens 4096 --resume
echo "[exit $?] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_formats2.py $O/run_qwen3_4b_2507 --gold train | tee $O/score_qwen3_4b_2507.txt
for i in 1 2; do
  echo "[start hf_repro_run$i] $(date -Is)"
  env PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 .venv/bin/python scripts/prompt_sweep.py --engine hf --model Qwen/Qwen3.5-4B --records $O1 --arm O1 --variants "$V" \
    --out-dir $O/hf_repro_run$i --max-new-tokens 4096 --limit 50 --max-batch 1 --resume
  echo "[exit $?] $(date -Is)"
done
echo "[all done] $(date -Is)"
