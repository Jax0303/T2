#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The 15 gold_cell failures: the reader had the gold cell sentence alone in
context and still missed. Prints the sentence's own value beside the answer, so
a scoring-convention miss is separable from a reading miss. Runs no model.
Output: results/lookup_gap/ceiling15.md."""
import json
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "analysis")
from phase4_summary import em                                        # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT     # noqa: E402

VAL = re.compile(r"\bis\s+(.+?)\s*\.\s*$")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def num(s):
    m = NUM.findall(str(s))
    return float(m[0].replace(",", "")) if m else None


a = types.SimpleNamespace(dataset="hitab", data_dir="data/hitab", split="dev",
                          population="hitab_dev_lookup_all", cell_scheme="S3c",
                          rhb_question_types=[], rhb_em_only=False, mh_queries=400,
                          seed=42, max_queries=0)
C = load_corpus(a)
text = {(t, i, j): caption_sentence(C.title.get(t, ""), *C.cell_paths[n],
                                    template=STRUCTURAL_COMPACT)
        for n, (t, i, j) in enumerate(C.cell_owner)}

rows = [json.loads(l) for l in open("results/phase4/reader_records.jsonl")]
bad = [r for r in rows if r["pool"] == "hitab_lookup" and r["policy"] == "gold_cell"
       and not em(r["pred_parsed"], r["gold_answer"])]

out, cls = [], {}
for r in bad:
    g = tuple(json.loads(r["gold_cell"])[0])
    sent = text.get((g[0], g[1], g[2]), "")
    m = VAL.search(sent)
    cv = m.group(1) if m else None
    p, gv, c = num(r["pred_parsed"]), num(r["gold_answer"]), num(cv)
    if c is not None and p is not None and abs(p - c) < 1e-9:
        k = "sign" if gv is not None and abs(abs(c) - abs(gv)) < 1e-9 else "cell_ne_gold"
    elif c is not None and p is not None and abs(p - c * 100) < 1e-6:
        k = "percent_scale"
    elif c is not None and p is not None and abs(p - (100 - c)) < 1e-6:
        k = "complement_100_minus_x"
    else:
        k = "other_reader_error"
    cls[k] = cls.get(k, 0) + 1
    out.append({"gold": r["gold_answer"], "pred": r["pred_parsed"],
                "cell_value": cv, "class": k, "sentence": sent})

md = ["# gold_cell 조건 오답 15건 — 리더 상한을 막는 것", "",
      "출처 `results/phase4/reader_records.jsonl` (policy=gold_cell) + S3c 셀 문장 재생성.",
      "계측기 `analysis/ceiling_cases.py`. 모델 실행 없음.",
      "컨텍스트에 gold 셀 문장 하나만 있는 조건이다 (EM 0.9206, n=189).", "",
      "| 분류 | n |", "|---|---:|"]
md += [f"| {k} | {v} |" for k, v in sorted(cls.items(), key=lambda x: -x[1])]
md += ["", "| gold | 셀 문장의 값 | 모델 답 | 분류 |", "|---|---|---|---|"]
md += [f"| {o['gold']} | {o['cell_value']} | {o['pred']} | {o['class']} |" for o in out]
md += ["", "`sign` = 모델이 셀 값을 그대로 옮겼고 gold만 부호를 뺐다.",
       "`percent_scale` = 셀 0.165를 16.5%로 바꿔 답했다.",
       "`complement_100_minus_x` = 셀 값의 100 − x를 답했다.",
       "`other_reader_error` = 셀 문장 하나만 주고도 다른 수를 답했다."]
p = Path("results/lookup_gap/ceiling15.md")
p.write_text("\n".join(md) + "\n")
print("\n".join(md))
