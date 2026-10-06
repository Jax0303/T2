"""PREREG-2026-10-06-claim-chain.md C: 답변 단계에 계산식 실행을 더한 조건 — 모든 검색 단위에 같은 답변 단계.

LLM 은 본 방법의 cot 지시문에 더해 산술이면 'Program: <숫자만 쓴 파이썬 식>' 줄을 쓰고, 프로그램이 그 식을 계산해
최종 값으로 쓴다(식이 없거나 계산이 안 되면 LLM 의 'Final answer:' 값). 같은 출력에서 LLM 자신의 값(pred_llm)도 채점해
실행 효과를 같은 문항 안에서 짝지어 본다.

  --source ours      본 방법 검색 레코드(doc.context 셀 문장 줄들)             --records results/mh_arms/..._records.jsonl
  --source langchain LangChain 청크 레코드(context 문자열)                   --records .../langchain_chunk_records.jsonl
  --source raptor    RAPTOR 레코드(context_<budget>)                          --records .../raptor_records.jsonl --budget 2000|575
채점은 공식 EM(mh_exact_match)·DocMath·형식 허용(format_match) 셋 다 적는다. 문항 단위로 저장하고 --resume 으로 잇는다.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_accuracy import check_context_limit                          # noqa: E402
from answer_accuracy_mh import COT, extract                                # noqa: E402
from mh_arms import official_answers                                       # noqa: E402
from rag_agent.eval.artifacts import digest, file_digest, provenance       # noqa: E402
from rag_agent.eval.multihiertt_em import (docmath_match, format_match,    # noqa: E402
                                           mh_exact_match)
from rag_agent.llm.factory import build_llm                                # noqa: E402

POT = (COT + " If the final value requires arithmetic, first write the exact computation as one "
       "Python arithmetic expression using only the numbers from the context (no variables, no units) "
       "on a line 'Program: ' and then the 'Final answer: ' line.")
_PROGRAM = re.compile(r"program:\s*(.+)", re.I)
_OPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Pow}


def run_program(text: str):
    """마지막 'Program:' 줄의 산술식을 계산한 값(문자열) 또는 None. 숫자·+-*/·괄호·단항 부호·**(지수 10 이하)만 허용.
    천 단위 쉼표·$·% 는 지운다. 그 밖(이름·호출·비교)은 실행하지 않는다."""
    hits = _PROGRAM.findall(text or "")
    if not hits:
        return None, None
    expr = hits[-1].strip().rstrip(".").strip("`").strip()
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr).replace("$", "").replace("%", "").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr, None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if abs(r) > 10: raise ValueError("exponent too large")
            return l ** r
        raise ValueError(f"disallowed node {type(node).__name__}")
    try:
        v = ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return expr, None
    if v != v or v in (float("inf"), float("-inf")):
        return expr, None
    return expr, f"{round(v, 6):.6f}".rstrip("0").rstrip(".")


def load_contexts(a):
    """[(query_id, kind, question, context 문자열, retrieval_correct)] — 소스별 레코드 형식."""
    rows = []
    for l in open(a.records, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if a.source == "ours":
            if "doc" not in r:
                continue
            ctx, ok = "\n".join(r["doc"]["context"]), r["doc"]["correct"]
        elif a.source == "langchain":
            ctx, ok = r["context"], r["correct"]
        else:
            if "excluded" in r:
                continue
            ctx, ok = r[f"context_{a.budget}"], r[f"correct_{a.budget}"]
        rows.append((r["query_id"], r["kind"], r["question"], ctx, int(ok)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["ours", "langchain", "raptor"])
    ap.add_argument("--records", required=True)
    ap.add_argument("--budget", type=int, default=2000, help="raptor: context_<budget>")
    ap.add_argument("--ids", default="", help="이 jsonl 의 query_id 만(답변 200문항). 없으면 레코드 전부")
    ap.add_argument("--split", default="train")
    ap.add_argument("--kinds", default="", help="lookup,arith 중 일부만(검증 집합 시험용)")
    ap.add_argument("--limit", type=int, default=0, help="query_id 순 앞 N 건")
    ap.add_argument("--reader", default="local:Qwen/Qwen3-8B?quantization=4bit")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix(".json").exists():
        raise SystemExit(f"{out.with_suffix('.json')} 있음 — 끝난 실행은 다시 돌리지 않는다")
    if out.exists() and not a.resume:
        raise SystemExit(f"{out} 있음 — 이어 쓰려면 --resume")

    rows = sorted(load_contexts(a), key=lambda x: x[0])
    if a.ids:
        want = {json.loads(l)["query_id"] for l in open(a.ids, encoding="utf-8") if l.strip()}
        rows = [x for x in rows if x[0] in want]
        if len(rows) != len(want):
            raise SystemExit(f"레코드에 없는 문항 {len(want) - len(rows)}건")
    if a.kinds:
        rows = [x for x in rows if x[1] in set(a.kinds.split(","))]
    if a.limit:
        rows = rows[:a.limit]
    answers = official_answers(a.split)
    done = {}
    if out.exists():
        done = {r["query_id"]: r for r in map(json.loads, open(out, encoding="utf-8")) if r}
        print(f"[resume] 저장된 {len(done)}건 건너뜀", flush=True)
    todo = [x for x in rows if x[0] not in done]
    print(f"[plan] {a.source} {len(rows)} 문항, 남은 {len(todo)}", flush=True)

    import torch
    llm = build_llm(a.reader)
    torch.manual_seed(a.seed)
    limit = llm.context_limit
    saved = [done[x[0]] for x in rows if x[0] in done]
    t0 = time.time()
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        for s in range(0, len(todo), a.batch_size):
            batch, t = todo[s:s + a.batch_size], time.time()
            users = [f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:" for _, _, q, ctx, _ in batch]
            n_toks = [llm.n_prompt_tokens(POT, u) for u in users]
            for n in n_toks:
                check_context_limit(n, a.max_tokens, limit)
            raws = ([llm.complete(POT, users[0], max_tokens=a.max_tokens, temperature=0.0)] if a.batch_size == 1
                    else llm.complete_batch(POT, users, max_tokens=a.max_tokens))
            sec = round((time.time() - t) / len(batch), 3)
            for (qid, kind, q, ctx, ok), n, raw in zip(batch, n_toks, raws):
                gold, prog = answers[qid], kind == "arith"
                pred_llm, marked = extract(raw)
                expr, pred_prog = run_program(raw)
                pred = pred_prog if pred_prog is not None else pred_llm
                row = {"query_id": qid, "kind": kind, "question": q, "retrieval_correct": ok,
                       "pred": pred, "pred_llm": pred_llm, "program": expr, "program_value": pred_prog,
                       "used_program": pred_prog is not None, "answer": gold,
                       "answer_correct": int(mh_exact_match(pred, gold, prog)),
                       "answer_correct_docmath": int(docmath_match(pred, gold, prog)),
                       "answer_correct_lenient": int(format_match(pred, gold, prog)),
                       "llm_correct": int(mh_exact_match(pred_llm, gold, prog)),
                       "llm_correct_lenient": int(format_match(pred_llm, gold, prog)),
                       "marker_found": marked, "n_tok": n, "raw": raw, "context_sha256": digest(ctx), "seconds": sec}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                saved.append(row)
            stream.flush(); os.fsync(stream.fileno())
            print(f"  {len(saved)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    by = defaultdict(list)
    for r in saved:
        by[r["kind"]].append(r); by["ALL"].append(r)
    rate = lambda k, v: round(sum(x[k] for x in v) / len(v), 4) if v else None
    summary = {"source": a.source, "records": a.records, "budget": a.budget if a.source == "raptor" else None,
               "ids": a.ids, "kinds": a.kinds, "limit": a.limit, "n": len(saved), "reader": llm.name,
               "prompt_text": POT, "prompt_sha256": digest(POT), "max_new_tokens": a.max_tokens, "batch_size": a.batch_size,
               "seed": a.seed, "temperature": 0.0, "program_found": sum(r["program"] is not None for r in saved),
               "program_executed": sum(r["used_program"] for r in saved), "marker_missing": sum(not r["marker_found"] for r in saved),
               "by_kind": {k: {"n": len(v), "em": rate("answer_correct", v), "lenient": rate("answer_correct_lenient", v),
                               "docmath": rate("answer_correct_docmath", v), "llm_em": rate("llm_correct", v),
                               "llm_lenient": rate("llm_correct_lenient", v), "program_executed": sum(x["used_program"] for x in v),
                               "retrieval_correct": rate("retrieval_correct", v)} for k, v in sorted(by.items())},
               "provenance": provenance(ROOT), "records_sha256": file_digest(out)}
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "program_found", "program_executed", "marker_missing", "by_kind")},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
