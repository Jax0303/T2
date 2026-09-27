"""리더 처리량 측정(2026-09-27): 기존 문맥 대 행 확장 문맥, 문항 10개씩. 답변 문자열·정오는 저장하지 않는다(시간·토큰만).

실행:  cd rag-agent && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/reader_throughput_20260927/measure.py
출력:  measure.json (같은 폴더)

- 문항: random.Random(20260927).sample(정렬한 id, 10). MultiHiertt = 882건 답변 표본(본 방법 최종 규칙 행),
  HiTab = s3c 답변 300건.
- 기존 문맥: 각 답변 실행이 읽은 검색 records 의 문맥(MultiHiertt 는 context_sha256 을 답변 행과 대조).
- 행 확장 문맥: 검색된 셀이 속한 행의 비어 있지 않은 데이터 셀 전부를 retrieval_accuracy.subtable_context
  (열 머리글 경로 + 행 머리글 경로를 붙인 markdown 표)로 그린 것. MultiHiertt 는 셀 문장이 문서 안에서 유일하므로
  문맥 문장을 다시 만든 셀 문장과 맞춰 셀 좌표를 찾는다. 표 제목이 없는 MultiHiertt 는 빈 제목 줄을 뺀다.
- 리더·프롬프트·최대 토큰은 답변 실행과 같다. MultiHiertt 는 한 건씩(complete)과 10건 한 번에(complete_batch),
  HiTab 은 답변 실행과 같은 한 건씩만. 첫 생성 전에 짧은 예열 생성 1회(시간에서 뺌).
"""
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import torch                                                        # noqa: E402

import mh_arms as m                                                 # noqa: E402
import retrieval_accuracy as ra                                     # noqa: E402
from answer_accuracy_mh import PROMPTS                              # noqa: E402  (neutral, cot 포함)
from rag_agent.eval.artifacts import digest                         # noqa: E402
from rag_agent.llm.factory import build_llm                         # noqa: E402

OUT = Path(__file__).with_name("measure.json")
SEED, N = 20260927, 10


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


def sha(p):
    h = hashlib.sha256()
    with open(ROOT / p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def row_expand(cells, tabs, page_titles):
    """검색된 셀 -> 그 셀들이 속한 행의 데이터 셀 전부 -> 머리글 경로를 붙인 markdown 표 줄."""
    full = set()
    for tid, i, _ in cells:
        t = tabs[tid].table
        full |= {(tid, i, j) for j in range(t.n_cols) if str(t.data[i][j]).strip()}
    units = ra.subtable_context(sorted(full), tabs, "", page_titles, header_mode="path")
    lines = [ln for u in units for ln in u["text"].split("\n") if ln.strip() != "#"]
    return lines, len(full)


def user_msg(lines, question):
    return "Context:\n" + "\n".join(lines) + f"\n\nQuestion: {question}\nAnswer:"


def measure(llm, system, max_tokens, items, batch):
    """items: [(qid, user)] -> 문항별 입력 토큰·출력 토큰·초. 답변 문자열은 버린다."""
    llm.complete(system, items[0][1], max_tokens=8, temperature=0.0)          # 예열
    out = {}
    for q, u in items:
        t = time.time()
        raw = llm.complete(system, u, max_tokens=max_tokens, temperature=0.0)
        out[q] = {"n_input_tokens": llm.n_prompt_tokens(system, u),
                  "n_output_tokens": len(llm.tokenizer(raw)["input_ids"]), "seconds_batch1": round(time.time() - t, 3)}
        del raw
    res = {"per_question": out,
           "seconds_per_question_batch1": round(sum(v["seconds_batch1"] for v in out.values()) / len(out), 3),
           "n_input_tokens_mean": round(sum(v["n_input_tokens"] for v in out.values()) / len(out), 1)}
    if batch:
        t = time.time()
        raws = llm.complete_batch(system, [u for _, u in items], max_tokens=max_tokens)
        res["seconds_total_batch10"] = round(time.time() - t, 3)
        res["seconds_per_question_batch10"] = round(res["seconds_total_batch10"] / len(items), 3)
        res["n_output_tokens_batch10_mean"] = round(sum(len(llm.tokenizer(r)["input_ids"]) for r in raws) / len(raws), 1)
        del raws
    return res


out = {"seed": SEED, "n": N, "device": torch.cuda.get_device_name(0)}

# ------------------------------------------------------------------ MultiHiertt (본 방법 최종 규칙, 882건 표본)
CAP = "results/mh_arms/cap300_20260924"
run = json.loads((ROOT / CAP / "cell_uniq.run.json").read_text())
REC = "results/mh_arms/mh_train_cell_hv3.3u_none_doc_records.jsonl"
assert sha(REC) == run["records_sha256"]
rows = {r["query_id"]: r for r in jl(f"{CAP}/cell_uniq.jsonl")}
pick = random.Random(SEED).sample(sorted(rows), N)
rec = {}
for x in jl(REC):
    if x["query_id"] in pick:
        doc = x["doc"] if isinstance(x["doc"], dict) else eval(x["doc"])
        rec[x["query_id"]] = doc["context"] if isinstance(doc["context"], list) else eval(doc["context"])
assert all(digest(rec[q]) == rows[q]["context_sha256"] for q in pick)
_, docs, _ = m.load_population("train")
tables, _ = m.build_tables({u: docs[u] for u in pick}, run["header_rule"], run["label_rule"])
texts, covers, _, unit_tids, _ = ra.build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
where = {}
for txt, cov, tid in zip(texts, covers, unit_tids):
    key = (tid.split("::")[0], txt)
    assert key not in where, "문서 안에 같은 셀 문장"
    where[key] = next(iter(cov))
tabs = {t: SimpleNamespace(table=v.table, title="") for t, v in tables.items()}
mh_items = {"base": [], "rowexp": []}
cells_n = {}
for q in pick:
    cells = [where[(q, s)] for s in rec[q]]
    lines, n_cells = row_expand(cells, tabs, {})
    cells_n[q] = {"base_cells": len(cells), "rowexp_cells": n_cells}
    mh_items["base"].append((q, user_msg(rec[q], rows[q]["question"])))
    mh_items["rowexp"].append((q, user_msg(lines, rows[q]["question"])))
llm = build_llm(run["reader"])
out["multihiertt"] = {"reader": run["reader"], "prompt": run["prompt"], "max_new_tokens": run["max_new_tokens"],
                      "production_batch_size": run["batch_size"], "questions": pick, "cells": cells_n,
                      "production_seconds_per_question_882": round(
                          sum(r["seconds"] for r in rows.values()) / len(rows), 3)}
for cond, items in mh_items.items():
    out["multihiertt"][cond] = measure(llm, PROMPTS[run["prompt"]], run["max_new_tokens"], items, batch=True)
del llm                      # 다음 모델을 올리기 전에 GPU 에서 내린다
gc.collect()
torch.cuda.empty_cache()
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ HiTab (s3c 답변 300건)
S3 = {r["query_id"]: r for r in jl("results/s3c_answer_hitab300_20260926/rows.jsonl")}
pick_h = random.Random(SEED).sample(sorted(S3), N)
hrec = {x["query_id"]: x for x in jl("results/retrieval_accuracy/t_s3c_gold_labelabl_records.jsonl") if x["query_id"] in pick_h}
hcell = {x["query_id"]: x for x in jl("results/retrieval_accuracy/t_s3c_gold_labelabl_type_accuracy.jsonl") if x["query_id"] in pick_h}
page_titles = json.loads(ra.PAGE_TITLES.read_text()) if ra.PAGE_TITLES.exists() else {}
htabs = {}
h_items = {"base": [], "rowexp": []}
hcells_n = {}
for q in pick_h:
    base_lines = [ln for e in hrec[q]["context"] for ln in str(e).split("\n") if ln.strip()]
    cells = [tuple(c) for c in hcell[q]["retrieved_cell_ids"]]
    assert len(cells) == hrec[q]["cells_in_context"]
    for tid in {c[0] for c in cells}:
        htabs.setdefault(tid, ra.hg.load_table(tid, str(ROOT / "data/hitab")))
    lines, n_cells = row_expand(cells, htabs, page_titles)
    hcells_n[q] = {"base_cells": len(cells), "rowexp_cells": n_cells}
    h_items["base"].append((q, user_msg(base_lines, hrec[q]["question"])))
    h_items["rowexp"].append((q, user_msg(lines, hrec[q]["question"])))
HREADER = "local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit"
llm = build_llm(HREADER)
out["hitab"] = {"reader": HREADER, "prompt": "neutral", "max_new_tokens": 64, "production_batch_size": 1,
                "questions": pick_h, "cells": hcells_n, "production_seconds_per_question_300": round(256 / 300, 3)}
for cond, items in h_items.items():
    out["hitab"][cond] = measure(llm, PROMPTS["neutral"], 64, items, batch=False)
del llm

# ------------------------------------------------------------------ 예상 시간(행 확장 조건 1개, 생성 시간만)
H, M = out["hitab"]["rowexp"], out["multihiertt"]["rowexp"]
n_h, n_m = 300 + 300, 300 + 882
out["estimate_rowexp"] = {
    "hitab_questions": n_h, "multihiertt_questions": n_m,
    "hitab_hours_batch1": round(n_h * H["seconds_per_question_batch1"] / 3600, 2),
    "multihiertt_hours_batch1": round(n_m * M["seconds_per_question_batch1"] / 3600, 2),
    "multihiertt_hours_batch10": round(n_m * M["seconds_per_question_batch10"] / 3600, 2),
    "formula": "문항 수 × 측정한 문항당 초 / 3600. 모델 적재 시간 제외. HiTab 은 한 건씩, MultiHiertt 는 한 건씩·10건 묶음 두 값."}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k in ("estimate_rowexp",)}, ensure_ascii=False, indent=1))
