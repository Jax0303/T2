#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""유형별 셀 검색 정확도 — 셀 개수 예산 없음 (2026-09-15 사용자 지정).

질의의 gold 셀이 m 개면, 검색 순위 상위 m 개 셀 문장이 정확히 gold 집합일 때 1, 아니면 0.
단일 셀은 top-1 = gold. 유형은 scripts/retrieval_accuracy.query_type (HiTab 필드, 예산 무관).
순위는 v2 기록의 context_units 순서(= 검색기 안정 정렬 순서; gold_rank 와 대조해 확인).
셀 단위 순위가 있는 표현(본 방법·MT2Net)만 이 규칙을 적용할 수 있다.

보정 EM: 검색 성공 질의에 그 m 개 정답 셀 문장만 준 답변 — 이미 있는
s3c_v2_answer_gold_modeall 기록을 읽고, 받은 문장이 검색된 문장과 같은지 해시로 확인한다.

  PYTHONPATH=. .venv/bin/python analysis/type_accuracy_v2.py
"""
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.retrieval_accuracy import TYPES, query_type               # noqa: E402
from rag_agent.eval.artifacts import digest, file_digest, read_records  # noqa: E402

D = ROOT / "results/evaluation_v2"
ARMS = {"s3c_v2": "본 방법 — 셀 문장(표 제목+행경로+열경로+값)",
        "mt2net_v2": "MT2Net 셀 단위(계층 헤더, 표 제목 없음)"}
OUT = D / "type_topm_exact_v2.json"


def rate(k, n):
    return round(k / n, 4) if n else None


def score(tag):
    path = D / f"{tag}_records.jsonl"
    assert json.loads((D / f"{tag}.json").read_text())["records_sha256"] == file_digest(path), path
    verdict, typ, excluded = {}, {}, 0
    for qid, r in read_records(path).items():
        if "excluded" in r:
            excluded += 1
            continue
        q = {"mode": r["mode"], "aggregation": r["aggregation"], "gold": r["gold_cells"]}
        if query_type(q) not in TYPES:
            continue
        gold = {tuple(c) for c in r["gold_cells"]}
        units = r["context_units"]
        assert len(gold) == r["m"] <= len(units) and all(len(u["cells"]) == 1 for u in units), qid
        first = next((i for i, u in enumerate(units, 1) if tuple(u["cells"][0]) in gold), None)
        assert first == r["gold_rank"] or first is None, qid     # 기록 순서 = 검색 순위
        verdict[qid] = int({tuple(u["cells"][0]) for u in units[:r["m"]]} == gold)
        typ[qid] = query_type(q)
    table = {}
    for t in (*TYPES, "overall"):
        ids = [q for q in verdict if t == "overall" or typ[q] == t]
        table[t] = {"n": len(ids), "success": sum(verdict[q] for q in ids),
                    "accuracy": rate(sum(verdict[q] for q in ids), len(ids))}
    table["n_excluded_unresolved_gold"] = excluded
    return table, verdict, typ, file_digest(path)


def main():
    scored = {tag: score(tag) for tag in ARMS}
    ours, ours_v, ours_t, _ = scored["s3c_v2"]
    _, mt_v, mt_t, _ = scored["mt2net_v2"]
    assert ours_t == mt_t

    tests, raw = {}, {}
    for t in (*TYPES, "overall"):
        ids = [q for q in ours_v if t == "overall" or ours_t[q] == t]
        w = sum(ours_v[q] and not mt_v[q] for q in ids)
        l = sum(mt_v[q] and not ours_v[q] for q in ids)
        tests[t] = {"wins": w, "losses": l, "p_exact": binomtest(w, w + l).pvalue if w + l else 1.0}
        if t != "overall":
            raw[t] = tests[t]["p_exact"]
    run = 0.0
    for i, t in enumerate(sorted(raw, key=raw.get)):
        run = max(run, min(1.0, (len(raw) - i) * raw[t]))
        tests[t]["p_holm_3"] = run

    rec_path = D / "s3c_v2_records.jsonl"
    gold_path = D / "s3c_v2_answer_gold_modeall.jsonl"
    assert json.loads(gold_path.with_suffix(".json").read_text())["retrieval_records_sha256"] == file_digest(rec_path)
    recs, gold = read_records(rec_path), read_records(gold_path)
    mismatch = []
    for qid in (q for q in ours_v if ours_v[q]):
        text = {tuple(u["cells"][0]): u["text"] for u in recs[qid]["context_units"][:recs[qid]["m"]]}
        ctx = [text[tuple(c)] for c in recs[qid]["gold_cells"]]
        if gold[qid]["context_condition"] != "gold" or digest(ctx) != gold[qid]["context_sha256"]:
            mismatch.append(qid)
    cal = {}
    for t in (*TYPES, "overall"):
        ids = [q for q in ours_v if t == "overall" or ours_t[q] == t]
        hit = [q for q in ids if ours_v[q]]
        ok = sum(gold[q]["answer_correct"] for q in hit)
        cal[t] = {"n": len(ids), "n_retrieval_success": len(hit),
                  "em_on_success_gold_sentences_only": rate(ok, len(hit)),
                  "em_over_n_failure_as_0": rate(ok, len(ids))}

    result = {"rule": "success iff set(top-m ranked cell sentences) == gold cells, m = |gold|; no cell budget",
              "sources": {f"{tag}_records.jsonl": scored[tag][3] for tag in ARMS}
                         | {gold_path.name: file_digest(gold_path)},
              "retrieval": {tag: {"label": ARMS[tag], "type_accuracy": scored[tag][0]} for tag in ARMS},
              "ours_vs_mt2net": tests,
              "not_applicable": "chunk1000_v2, huawei_char_v2, rowcol_v2: units carry many cells, no per-cell ranking",
              "calibration_ours": cal,
              "calibration_context_order": "gold_cells order (sorted coordinates), not rank order",
              "gold_context_not_equal_retrieved_top_m_sentences": mismatch}
    with OUT.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    for tag in ARMS:
        ta = scored[tag][0]
        print(f"| {tag} | " + " | ".join(f"{ta[t]['success']}/{ta[t]['n']} = {ta[t]['accuracy']:.4f}"
                                         for t in (*TYPES, "overall")) + " |")
    for t, v in tests.items():
        print(f"ours vs mt2net {t}: {v['wins']}:{v['losses']} p={v['p_exact']:.3g} holm={v.get('p_holm_3', float('nan')):.3g}")
    for t, v in cal.items():
        print(f"calib {t}: n={v['n']} success={v['n_retrieval_success']} "
              f"EM|success={v['em_on_success_gold_sentences_only']} EM(fail=0)={v['em_over_n_failure_as_0']}")
    print(f"gold context != retrieved top-m sentences: {len(mismatch)}\nwrote {OUT}")


if __name__ == "__main__":
    main()
