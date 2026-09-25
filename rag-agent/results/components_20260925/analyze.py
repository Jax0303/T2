"""2026-09-25 구성요소 비교 집계: 검색 정확도·정답 표 포함·McNemar(e 대비, Holm 4개),
문장 중복률, s3c 실패 분해. 모집단 = HiTab test 단일 셀 조회(mode all, m=1, 집계 없음).
실행: .venv/bin/python results/components_20260925/analyze.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import retrieval_accuracy as ra                                       # noqa: E402
from rag_agent.eval.artifacts import digest                           # noqa: E402

OUT = Path(__file__).parent
FORMS = [  # (이름, 템플릿, 검색 레코드, 요약 JSON)
    ("a", "value", OUT / "value_records.jsonl", OUT / "value.json"),
    ("b", "flat", OUT / "flat_records.jsonl", OUT / "flat.json"),
    ("c", "s2", ROOT / "results/retrieval_accuracy/t_s2_split_labelabl_records.jsonl",
     ROOT / "results/retrieval_accuracy/t_s2_split_labelabl.json"),
    ("d", "s3label", OUT / "s3label_records.jsonl", OUT / "s3label.json"),
    ("e", "s3c", ROOT / "results/evaluation_v2/s3c_v2_records.jsonl",
     ROOT / "results/evaluation_v2/s3c_v2.json"),
]


def primary(path):
    return {r["query_id"]: r for r in map(json.loads, open(path))
            if "excluded" not in r and r["mode"] == "all" and r["m"] == 1
            and (r.get("aggregation") or "none") == "none"}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


recs = {k: primary(p) for k, _, p, _ in FORMS}
qids = sorted(recs["e"])
assert len(qids) == 991 and all(set(r) == set(qids) for r in recs.values())
gold = {q: tuple(recs["e"][q]["gold_cells"][0]) for q in qids}

# 문장 중복: 인덱스와 같은 코퍼스를 다시 만들어 해시가 요약 JSON 과 같은지 확인한다
page_titles = json.loads(ra.PAGE_TITLES.read_text())
queries = ra.load_queries("data/hitab", "test", {})
tids = sorted({q["table_id"] for q in queries})
result = {"forms": {}, "mcnemar_vs_e": {}, "s3c_failures": {}}
dup_of_gold = {}
for k, tpl, rpath, jpath in FORMS:
    texts, covers, *_ = ra.build_corpus("data/hitab", tids, tpl, "cell", page_titles)
    assert digest(texts) == json.loads(Path(jpath).read_text())["corpus_text_sha256"], tpl
    cnt = Counter(texts)
    in_table = Counter((t, next(iter(c))[0]) for t, c in zip(texts, covers))
    text_of = {next(iter(c)): t for t, c in zip(texts, covers)}
    n_dup = sum(1 for t in texts if cnt[t] > 1)
    # 정답 셀 문장이 다른 셀과 같은가 — 같은 표 안의 셀과 같은가 / 다른 표의 셀과만 같은가
    g = {}
    for q in qids:
        t = text_of[gold[q]]
        g[q] = ("none" if cnt[t] == 1 else
                "same_table" if in_table[t, gold[q][0]] > 1 else "other_table_only")
    dup_of_gold[k] = g
    r = recs[k]
    result["forms"][k] = {
        "template": tpl, "records": str(Path(rpath).relative_to(ROOT)),
        "example": text_of[("100", 3, 0)],
        "n_queries": len(qids), "correct": sum(r[q]["correct"] for q in qids),
        "gold_table_in_context": sum(r[q]["gold_table_in_context"] for q in qids),
        "corpus_cells": len(texts), "cells_with_duplicate_text": n_dup,
        "distinct_texts": len(cnt),
        "queries_gold_text_duplicated": sum(v != "none" for v in g.values()),
        "queries_gold_dup_same_table": sum(v == "same_table" for v in g.values()),
        "queries_gold_dup_other_table_only": sum(v == "other_table_only" for v in g.values()),
    }
    print(k, tpl, result["forms"][k], flush=True)

ps = []
for k in "abcd":
    b = sum(recs[k][q]["correct"] and not recs["e"][q]["correct"] for q in qids)
    c = sum(recs["e"][q]["correct"] and not recs[k][q]["correct"] for q in qids)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    ps.append(p)
    result["mcnemar_vs_e"][k] = {"only_this_correct": b, "only_e_correct": c, "p_exact": p}
for k, adj in zip("abcd", holm(ps)):
    result["mcnemar_vs_e"][k]["p_holm"] = adj

e = recs["e"]
fails = [q for q in qids if not e[q]["correct"]]
for name, sel in (("table_missed", [q for q in fails if not e[q]["gold_table_in_context"]]),
                  ("table_found_cell_missed", [q for q in fails if e[q]["gold_table_in_context"]])):
    result["s3c_failures"][name] = {
        "n": len(sel),
        "gold_text_duplicated": sum(dup_of_gold["e"][q] != "none" for q in sel),
        "dup_same_table": sum(dup_of_gold["e"][q] == "same_table" for q in sel),
        "dup_other_table_only": sum(dup_of_gold["e"][q] == "other_table_only" for q in sel),
    }
(OUT / "components.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: result[k] for k in ("mcnemar_vs_e", "s3c_failures")}, indent=2))
