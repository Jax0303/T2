"""본 방법 셀 문장 20개 + LangChain 튜토리얼 지시문·틀 — PREREG-2026-09-30-cell-tutorial-prompt.md.

입력 내용만 본 방법(cell_uniq 882 답변과 같은 셀 문장), 지시문·프롬프트 틀·system 없음은 langchain_chunk.py 그대로.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/baselines"))

from langchain_chunk import MARKER, MAX_NEW, PROMPT, RAG_PROMPT, READER, SEED, BATCH  # noqa: E402
from answer_accuracy import check_context_limit                        # noqa: E402
from answer_accuracy_mh import extract                                  # noqa: E402
from mh_arms import official_answers                                    # noqa: E402
from rag_agent.eval.artifacts import digest, file_digest, provenance, read_records  # noqa: E402
from rag_agent.eval.multihiertt_em import mh_exact_match, self_check   # noqa: E402

RECORDS = ROOT / "results/mh_arms/mh_train_cell_hv3.3u_none_doc_records.jsonl"
OURS = ROOT / "results/mh_arms/cap300_20260924/cell_uniq.jsonl"
IDS = ROOT / "results/mh_only_20260928/baselines/ids200.jsonl"
OUT = ROOT / "results/cell_tutorial_prompt_20260930"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", default=str(IDS), help="문항 목록 jsonl(query_id). 882 = results/mh_arms/cap300_20260924/cell_uniq.jsonl")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--resume", action="store_true", help="answer.jsonl 의 저장된 행은 건너뛰고 이어 쓴다")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    import torch
    from rag_agent.llm.local_qwen import LocalQwenLLM

    class UserOnly(LocalQwenLLM):
        """langchain_chunk.py 와 같다 — human 메시지 하나, system 없음."""
        def _prompt(self, system, user, thinking=False):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)

    recs = read_records(RECORDS)
    ours = {r["query_id"]: r for r in map(json.loads, open(OURS, encoding="utf-8"))}
    want = [json.loads(l)["query_id"] for l in open(a.ids, encoding="utf-8") if l.strip()]
    rows_in = [recs[q] for q in want]
    bad = [r["query_id"] for r in rows_in if digest(r["doc"]["context"]) != ours[r["query_id"]]["context_sha256"]]
    if bad:
        raise SystemExit(f"셀 문장이 본 방법 답변 실행과 다르다: {len(bad)}건 (예: {bad[:3]})")
    print(f"[same-contexts] {len(rows_in)}건 context_sha256 일치", flush=True)
    answers = official_answers("train")
    check = self_check([answers[r["query_id"]] for r in rows_in], [r["kind"] == "arith" for r in rows_in])
    print(f"[scorer] gold 를 예측으로 넣었을 때 EM={check['em']} (n={check['n']})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "answer.jsonl"
    done = {}
    if out.exists():
        if not a.resume:
            raise SystemExit(f"{out} 있음 — 덮어쓰지 않는다(--resume)")
        done = {r["query_id"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
        print(f"[resume] 저장된 {len(done)}건 건너뜀", flush=True)
    llm = UserOnly(model_name=READER, quantization="4bit")
    torch.manual_seed(SEED)
    rows, t0 = [done[r["query_id"]] for r in rows_in if r["query_id"] in done], time.time()
    pending = [r for r in rows_in if r["query_id"] not in done]
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        for s in range(0, len(pending), BATCH):
            batch, t = pending[s:s + BATCH], time.time()
            users = [PROMPT.format(instr=RAG_PROMPT + MARKER, question=r["question"],
                                   context="\n".join(r["doc"]["context"])) for r in batch]
            n_toks = [llm.n_prompt_tokens("", u) for u in users]
            for n in n_toks:
                check_context_limit(n, MAX_NEW, llm.context_limit)
            raws = llm.complete_batch("", users, max_tokens=MAX_NEW)
            sec = round((time.time() - t) / len(batch), 3)
            for r, n, raw in zip(batch, n_toks, raws):
                gold = answers[r["query_id"]]
                pred, marked = extract(raw)
                row = {"query_id": r["query_id"], "layer": r["layer"], "kind": r["kind"], "m": r["m"],
                       "question": r["question"], "retrieval_correct": r["doc"]["correct"],
                       "pred": pred, "answer": gold,
                       "answer_correct": int(mh_exact_match(pred, gold, r["kind"] == "arith")),
                       "raw": raw, "marker_found": marked, "n_tok": n,
                       "context_sha256": digest(r["doc"]["context"]), "seconds": sec}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            stream.flush(); os.fsync(stream.fileno())
            print(f"  {len(rows)}/{len(rows_in)}  {time.time() - t0:.0f}s", flush=True)
    by = defaultdict(list)
    for r in rows:
        by[r["layer"]].append(r["answer_correct"]); by["ALL"].append(r["answer_correct"])
    summary = {"prompt_text": PROMPT.format(instr=RAG_PROMPT + MARKER, question="{question}", context="{context}"),
               "system_message": None, "reader": llm.name, "max_new_tokens": MAX_NEW, "seed": SEED,
               "batch_size": BATCH, "n": len(rows), "marker_missing": sum(not x["marker_found"] for x in rows),
               "input_tokens_mean": round(sum(x["n_tok"] for x in rows) / len(rows), 1),
               "by_layer": {k: {"n": len(v), "correct": sum(v)} for k, v in sorted(by.items())},
               "scorer_self_check": check, "records_sha256": file_digest(out), "provenance": provenance(ROOT)}
    (out_dir / "answer.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
