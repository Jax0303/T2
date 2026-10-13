"""GPT-4o 확인 실험 입력 만들기 — PREREG-2026-10-13-gpt4o.md (실행 전 생성).

[표 전부] fulldoc_records.jsonl : 검색 없이 질문의 문서에 있는 표 전부를 LangChain 청킹 원문과 같은 markdown 으로(`langchain_chunk.doc_text`).
[질문만] closed_records.jsonl  : 표 없이 질문만(context 빈 문자열, `--prompt closed` 와 같이 쓴다).
두 조건 모두 문항·순서·기본 필드는 셀 문장 분해 실험의 C_ours_budget_records.jsonl(882) 에서 그대로 가져온다.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/gpt4o_20261013/build_inputs.py
"""
import json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/baselines")]
from langchain_chunk import doc_text                                 # noqa: E402
from mh_arms import build_tables, load_population                    # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

OUT = ROOT / "results/gpt4o_20261013"
SRC = ROOT / "results/cellform_20261012/C_ours_budget_records.jsonl"
BASE = ("query_id", "kind", "layer", "m", "question", "gold_ids")

src = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
assert len(src) == 882
_q, docs, _ = load_population("train")
tables, hdr = build_tables({r["query_id"]: docs[r["query_id"]] for r in src}, "v3.3u", "none")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

full, closed = [], []
for r in src:
    base = {k: r[k] for k in BASE}
    text, spans = doc_text(r["query_id"], tables, hdr)
    cells = {c for _s, _e, c in spans}
    assert set(r["gold_ids"]) <= cells, r["query_id"]                 # 정답 셀은 문서의 표 안에 있어야 한다
    full.append({**base, "context": text, "correct": 1, "cells_in_context": len(cells), "n_tok_ctx": n_tok(text)})
    closed.append({**base, "context": "", "correct": 0, "cells_in_context": 0, "n_tok_ctx": 0})

for name, rows in (("fulldoc", full), ("closed", closed)):
    p = OUT / f"{name}_records.jsonl"
    if p.exists():
        raise SystemExit(f"{p} 있음 — 덮어쓰지 않는다")
    with p.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    v = [x["n_tok_ctx"] for x in rows]
    c = [x["cells_in_context"] for x in rows]
    print(f"{name}: {len(rows)}건, 정답 셀 포함 {sum(x['correct'] for x in rows)}, 셀 평균 {st.mean(c):.1f}, "
          f"토큰 평균 {st.mean(v):,.0f} 중앙값 {st.median(v):,.0f} 최대 {max(v):,} 합계 {sum(v):,}")
