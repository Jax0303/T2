#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""리더에 들어가는 입력 크기 — 토크나이저만 쓰고 생성은 하지 않는다.

조건마다 프롬프트가 얼마나 크고 문맥 한계를 넘는 질의가 몇 건인지 잰다. 생성이
없으므로 GPU 도 예측값도 필요 없고, 그래서 **본실행 전에** 돌려 OOM 위험 지점을
먼저 안다. 프롬프트는 `scripts/answer_accuracy.py` 가 만드는 것과 같은 문자열이다
— 같은 함수를 불러 쓰므로 두 벌이 되지 않는다.

입력은 어디서도 잘리지 않는다(`local_qwen.complete` 에 truncation 이 없다). 그래서
아래 `n_over_limit` 은 절단 건수가 아니라 **실패할 건수**다. 0 이 아니면 그 조건은
돌릴 수 없다.

  PYTHONPATH=. .venv/bin/python analysis/answer_input_size.py
"""
from __future__ import annotations

import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_s = importlib.util.spec_from_file_location("aa", ROOT / "scripts/answer_accuracy.py")
aa = importlib.util.module_from_spec(_s)
_s.loader.exec_module(aa)

D = ROOT / "results/retrieval_accuracy"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

ARMS = [("본 방법 (셀+헤더경로)", "t_s3c_hybrid"),
        ("행 단위 (RowColRetrieval 행 절반)", "t_row_values"),
        ("RowColRetrieval (행×열 온전)", "t_rowcol_values"),
        ("MT2Net 색인 단위", "t_mt2net_hybrid"),
        ("TableRAG Huawei (코드 1,000자)", "t_trag_hetero"),
        ("TableRAG Huawei (논문 1,000토큰)", "t_trag_hetero_tok"),
        ("TableRAG NeurIPS'24 (leaf)", "t_tablerag_leaf"),
        ("TableRAG NeurIPS'24 (path)", "t_tablerag_path"),
        ("고정 청킹 1,000자 (업계 기본값)", "t_chunk1000")]
#: `표 통째` 는 답변 표에서 뺀다 (2026-09-09 사용자 결정). 표 1 에서도 셀 검색
#: 단계가 없어 주지표 칸이 비어 있는 통제 행이고, 문맥이 평균 118.6셀 / 최대
#: 13,729토큰이라 8GB 에서 6.2% 의 질의가 VRAM 절벽에 걸려 10시간이 든다.
#: 측정하지 않았으므로 표에 행을 만들지 않는다.
#: 표 1b — 우리 ablation. 경쟁 상대가 아니므로 본표에서 뺀다.
ABLATION = [("우리 문장을 행별로 묶음", "t_row_hybrid")]


def measure(tag: str, tok, limit: int, prompt: str = "base") -> dict:
    f = D / f"{tag}_records.jsonl"
    recs = [j for j in map(json.loads, f.open()) if "correct" in j]
    toks, cells = [], []
    for r in recs:
        ctx = r.get("context") or []
        user = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {r['question']}\nAnswer:"
        msgs = [{"role": "system", "content": aa.PROMPTS[prompt]},
                {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        toks.append(len(tok(text)["input_ids"]))
        cells.append(r["cells_in_context"])
    over = [t for t in toks if t > limit]
    return {"n": len(recs), "tokens_mean": round(st.mean(toks), 1),
            "tokens_median": st.median(toks), "tokens_max": max(toks),
            "cells_mean": round(st.mean(cells), 1), "cells_max": max(cells),
            "context_limit": limit, "n_over_limit": len(over),
            "over_limit_ratio": round(len(over) / len(recs), 4)}


def main() -> int:
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    limit = int(AutoConfig.from_pretrained(MODEL).max_position_embeddings)
    out = {"model": MODEL, "prompt": "base", "conditions": {}}
    for name, tag in ARMS + ABLATION:
        if not (D / f"{tag}_records.jsonl").exists():
            print(f"{name}: {tag} 레코드 없음 — 건너뜀", flush=True)
            continue
        m = measure(tag, tok, limit)
        out["conditions"][name] = dict(arm=tag, **m)
        print(f"{name:16} n={m['n']} 셀 {m['cells_mean']:6.1f} "
              f"토큰 평균 {m['tokens_mean']:8.1f} 중앙 {m['tokens_median']:6d} "
              f"최대 {m['tokens_max']:6d} 한계초과 {m['n_over_limit']}", flush=True)
    (D / "INPUT_SIZE.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {D / 'INPUT_SIZE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
