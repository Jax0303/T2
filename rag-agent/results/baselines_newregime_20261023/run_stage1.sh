#!/usr/bin/env bash
# 1단계(vendor 불필요): 표 전체 입력 P*(vLLM) → LangChain 답변(HF, tutorial 지시문) — PREREG-2026-10-23-baselines-newregime.md
set -uo pipefail
cd "$(dirname "$0")/../.."
O=results/baselines_newregime_20261023; M=Qwen/Qwen3.5-4B; V=R1X1F1B2D1C1E0; R="local:Qwen/Qwen3.5-4B?quantization=none"
F=results/gpt4o_20261013/fulldoc_records.jsonl
ENV="CUDA_HOME=$HOME/vllm-venv/lib/python3.12/site-packages/nvidia/cu13 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
MML=$(env PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 .venv/bin/python - "$M" "$V" "$F" <<'PY'
import json, sys
from transformers import AutoTokenizer
import prompt_variants as pv
M, V, f = sys.argv[1:]
tok = AutoTokenizer.from_pretrained(M); m = 0
for l in open(f, encoding="utf-8"):
    r = json.loads(l)
    p = tok.apply_chat_template([{"role": "user", "content": pv.build(V, r["question"], r["context"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    m = max(m, len(tok(p).input_ids))
print("max prompt tokens", m, file=sys.stderr)
print(8192 if m + 4200 <= 8192 else (10240 if m + 4200 <= 10240 else (12288 if m + 4200 <= 12288 else 16384)))
PY
)
echo "[start fulldoc P* seqs=16 max-model-len=$MML] $(date -Is)"
env $ENV ~/vllm-venv/bin/python scripts/prompt_sweep.py --engine vllm --model $M --gpu-mem 0.89 --max-model-len $MML --max-num-seqs 16 \
  --records $F --arm F --variants "$V" --out-dir $O/run_fulldoc_pstar --max-new-tokens 4096 --resume
echo "[exit $? fulldoc] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/format_dev_20261017/score_formats2.py $O/run_fulldoc_pstar --gold train | tee $O/score_fulldoc_pstar.txt
echo "[start langchain answer HF] $(date -Is)"
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/baselines/langchain_chunk.py --stage answer --out-dir $O/langchain_chunk \
  --records results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl --reader "$R" --batch-size 16 --resume
echo "[exit $? langchain] $(date -Is)"
echo "[stage1 done] $(date -Is)"
