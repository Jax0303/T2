"""2026-09-26 HiTab s3c_v2 (리더 없음): 질문 단어 중 정답 셀 경로(행+열)에 나오는 단어 비율.
모집단 = test 단일 셀 조회 991문항 중 '표는 찾았으나 셀 못 찾음' 45 대 '검색 성공' 906.
단어 = BM25 와 같은 토크나이저(rag_agent/retrieve/encoders._tokenize, 소문자, 어간 처리 없음), 서로 다른 단어만,
불용어 = sklearn ENGLISH_STOP_WORDS. 비율 = |질문 단어 ∩ 경로 단어| / |질문 단어|. 표 제목·값은 경로에 넣지 않는다.
실행: .venv/bin/python results/components_20260925/hitab_path_overlap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rag_agent.bench import hitab_grid as hg                          # noqa: E402
from rag_agent.retrieve.encoders import _tokenize                     # noqa: E402

words = lambda s: set(_tokenize(s)) - ENGLISH_STOP_WORDS
tables, rows = {}, []
for r in map(json.loads, open(ROOT / "results/evaluation_v2/s3c_v2_records.jsonl")):
    if "excluded" in r or r["mode"] != "all" or r["m"] != 1 or (r.get("aggregation") or "none") != "none":
        continue
    tid, i, j = r["gold_cells"][0]
    t = tables.setdefault(tid, hg.load_table(tid, "data/hitab").table)
    q, p = words(r["question"]), words(" ".join([*t.row_path(i), *t.col_path(j)]))
    rows.append({"query_id": r["query_id"], "correct": r["correct"], "table_found": r["gold_table_in_context"],
                 "n_q_words": len(q), "n_in_path": len(q & p), "ratio": len(q & p) / len(q) if q else None})
assert len(rows) == 991
groups = {"fail_table_found": [x for x in rows if not x["correct"] and x["table_found"]],
          "success": [x for x in rows if x["correct"]]}
assert (len(groups["fail_table_found"]), len(groups["success"])) == (45, 906)
out = {}
for k, g in groups.items():
    v = np.array([x["ratio"] for x in g if x["ratio"] is not None])
    out[k] = {"n": len(g), "n_no_content_words": sum(x["ratio"] is None for x in g),
              "ratio_mean": float(v.mean()), "ratio_median": float(np.median(v)),
              "ratio_zero": int((v == 0).sum()), "ratio_one": int((v == 1).sum()),
              "q_words_mean": float(np.mean([x["n_q_words"] for x in g])),
              "in_path_words_mean": float(np.mean([x["n_in_path"] for x in g]))}
a, b = ([x["ratio"] for x in groups[k] if x["ratio"] is not None] for k in groups)
out["mannwhitney_two_sided_p"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
(Path(__file__).parent / "hitab_path_overlap.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
