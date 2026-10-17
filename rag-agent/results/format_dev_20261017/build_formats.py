"""전달 형식 dev 입력 — PREREG-2026-10-17-delivery-format.md. 같은 문항·같은 검색 순위·같은 토큰 길이 T(= LangChain 청크 4개 원문 토큰)에서
  C  (기존) 점수 순 셀 문장 목록                                  — results/prompt_sweep_20261015/dev_C_records.jsonl 그대로(여기서 재현 확인만)
  G  같은 선택 규칙, 표별로 묶고(표는 가장 높은 순위 셀 순) 표 안은 (행, 열) 순, 표마다 'Table k' 한 줄
  S0 성긴 부분 표, 단 셀은 C 가 고른 셀 그대로(같은 셀·다른 모양 — 옛 [라] 조건의 재현)
  S  성긴 부분 표: 표별 markdown, 행 = 선택 셀의 행(표 순서), 열 = 선택 셀의 열(표 순서), 선택 셀만 값, 나머지 빈칸
  D  채운 부분 표: S 와 같되 행×열 교차 칸을 문서 값으로 전부 채움(빈 셀 제외)
  A  (기존) LangChain 청크 4개                                     — dev_A_records.jsonl 그대로
선택 규칙(전부 같음): 검색 상위 100 순서로 셀을 하나씩 더해 렌더링한 글의 토큰(Qwen3-8B 토크나이저)이 T 를 넘기 직전까지.
  PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/format_dev_20261017/build_formats.py
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from mh_arms import build_tables, cell_id, load_population          # noqa: E402
from retrieval_accuracy import build_corpus                          # noqa: E402
from transformers import AutoTokenizer                               # noqa: E402

OUT = Path(__file__).resolve().parent
SW = ROOT / "results/prompt_sweep_20261015"
R100 = {"validation": ROOT / "results/mh_only_20260928/dev_k/dev_s3c_k100_records.jsonl",
        "train": ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
A = {r["query_id"]: r for r in map(json.loads, open(SW / "dev_A_records.jsonl", encoding="utf-8"))}
C = {r["query_id"]: r for r in map(json.loads, open(SW / "dev_C_records.jsonl", encoding="utf-8"))}
ids = json.load(open(SW / "dev_ids.json"))
assert len(ids) == 395 and set(ids) == set(A) == set(C)
by_split = defaultdict(list)
for q in ids:
    by_split[A[q]["split"]].append(q)
TBL = re.compile(r"^Table \d+$")
esc = lambda v: str(v).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", "")
strip_tbl = lambda path: [p for k, p in enumerate(path) if not (k == 0 and TBL.match(str(p)))]

rows = {"S0": [], "G": [], "S": [], "D": []}
stats = Counter()
samples = []
for split, qs in by_split.items():
    _q, docs, _ = load_population(split)
    tables, hdr = build_tables({u: docs[u] for u in qs}, "v3.3u", "none")
    texts, covers, _, unit_tids, _ = build_corpus("", sorted(tables), "s3c", "cell", {}, load=lambda t, _d: tables.get(t))
    sent, coord = defaultdict(dict), defaultdict(dict)
    for txt, cov, tid in zip(texts, covers, unit_tids):
        c = next(iter(cov)); s = cell_id(c, hdr).split("::", 1)[1]
        sent[tid.split("::")[0]][s] = txt; coord[tid.split("::")[0]][s] = c
    r100 = {r["query_id"]: r for r in map(json.loads, open(R100[split], encoding="utf-8")) if r["query_id"] in set(qs) and "doc" in r}
    assert set(r100) == set(qs), (split, len(r100), len(qs))
    for q in qs:
        rank = [u[0] for u in r100[q]["doc"]["ranked_units"]]
        T = A[q]["n_tok_ctx"]; gold = set(A[q]["gold_ids"])
        S_, K_ = sent[q], coord[q]
        short = lambda tid, i, j: cell_id((tid, i, j), hdr).split("::", 1)[1]
        # C 재현 확인(같은 선택 규칙)
        sel = []
        for u in rank:
            if n_tok("\n".join(S_[c] for c in sel + [u])) > T: break
            sel.append(u)
        assert "\n".join(S_[c] for c in sel) == C[q]["context"], ("C 재현 실패", q)
        def grouped(sel_ids):
            order, per = [], {}
            for s in sel_ids:
                tid, i, j = K_[s]
                if tid not in per: order.append(tid); per[tid] = []
                per[tid].append((i, j, s))
            blocks = []
            for tid in order:
                k = int(tid.split("::")[1]) + 1
                blocks.append(f"Table {k}\n" + "\n".join(S_[s] for i, j, s in sorted(per[tid])))
            return "\n\n".join(blocks), set(sel_ids)
        def subtable(sel_ids, dense):
            order, per = [], {}
            for s in sel_ids:
                tid, i, j = K_[s]
                if tid not in per: order.append(tid); per[tid] = {"rows": set(), "cols": set(), "cells": set()}
                per[tid]["rows"].add(i); per[tid]["cols"].add(j); per[tid]["cells"].add((i, j))
            blocks, delivered = [], set()
            for tid in order:
                t = tables[tid].table; rws = sorted(per[tid]["rows"]); cls = sorted(per[tid]["cols"]); k = int(tid.split("::")[1]) + 1
                lines = [f"Table {k}", "| | " + " | ".join(esc(" > ".join(strip_tbl(t.col_path(j)))) for j in cls) + " |", "| --- | " + " | ".join("---" for _ in cls) + " |"]
                for i in rws:
                    vals = []
                    for j in cls:
                        v = t.data[i][j]; show = (i, j) in per[tid]["cells"] or (dense and str(v).strip() != "")
                        vals.append(esc(v) if show else "")
                        if show and str(v).strip(): delivered.add(short(tid, i, j))
                    lines.append("| " + esc(" > ".join(strip_tbl(t.row_path(i)))) + " | " + " | ".join(vals) + " |")
                blocks.append("\n".join(lines))
            return "\n\n".join(blocks), delivered
        text0, deliv0 = subtable(sel, False)                       # S0: C 와 같은 셀, 성긴 부분 표(토큰은 C 보다 적다)
        rec0 = {k: A[q][k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids", "split", "answer")}
        rec0.update({"context": text0, "correct": int(gold <= deliv0), "cells_in_context": len(deliv0), "n_tok_ctx": n_tok(text0), "selected_units": len(sel), "budget_T": T})
        assert deliv0 == set(sel), q
        rows["S0"].append(rec0)
        for arm, fn in (("G", grouped), ("S", lambda s_: subtable(s_, False)), ("D", lambda s_: subtable(s_, True))):
            sel2, best = [], ("", set())
            for u in rank:
                text, deliv = fn(sel2 + [u])
                if n_tok(text) > T: break
                sel2.append(u); best = (text, deliv)
            text, deliv = best
            rec = {k: A[q][k] for k in ("query_id", "kind", "layer", "m", "question", "gold_ids", "split", "answer")}
            rec.update({"context": text, "correct": int(gold <= deliv), "cells_in_context": len(deliv), "n_tok_ctx": n_tok(text), "selected_units": len(sel2), "budget_T": T})
            rows[arm].append(rec)
        if len(samples) < 2 and A[q]["kind"] == "arith":
            samples.append((q, rows["G"][-1]["context"][:700], rows["S"][-1]["context"][:900], rows["D"][-1]["context"][:900]))
for arm in rows:
    rs = sorted(rows[arm], key=lambda r: ids.index(r["query_id"]))
    with open(OUT / f"{arm}_records.jsonl", "x", encoding="utf-8") as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{arm}: {len(rs)}건, 정답 셀 포함 {sum(r['correct'] for r in rs)}, 전달 셀 평균 {sum(r['cells_in_context'] for r in rs)/len(rs):.1f}, 선택 단위 평균 {sum(r['selected_units'] for r in rs)/len(rs):.1f}, 토큰 평균 {sum(r['n_tok_ctx'] for r in rs)/len(rs):.0f}, T 초과 {sum(r['n_tok_ctx'] > r['budget_T'] for r in rs)}")
cs = [C[q] for q in ids]; as_ = [A[q] for q in ids]
print(f"C: 정답 셀 포함 {sum(r['correct'] for r in cs)}, 셀 평균 {sum(r['cells_in_context'] for r in cs)/395:.1f}, 토큰 {sum(r['n_tok_ctx'] for r in cs)/395:.0f} | A: 포함 {sum(r['correct'] for r in as_)}, 셀 {sum(r['cells_in_context'] for r in as_)/395:.1f}, 토큰 {sum(r['n_tok_ctx'] for r in as_)/395:.0f}")
for q, g, s, d in samples:
    print(f"\n===== 예시 {q}\n--- G ---\n{g}\n--- S ---\n{s}\n--- D ---\n{d}")
