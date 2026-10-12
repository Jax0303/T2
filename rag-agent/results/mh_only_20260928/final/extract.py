"""줄인 비교군의 검색·답변 수치를 한 번에 뽑는다(2026-09-29). 새 실행 없음.

입력 파일은 아래 ARMS 에 경로를 직접 적은 것만 쓴다(비슷한 이름의 옛 파일을 고르지 않도록).
각 파일은 설정을 확인하고, 답변 파일은 문항마다 리더가 받은 문맥이 오늘 검색 기록의 문맥과 같은지 대조한다.
  검색: MultiHiertt train 2,885, 문서 안, 전달 셀 20 — 정답 셀 전부 포함 비율.
  답변: 같은 882 문항, Qwen3-8B cot 384, EM. 본 방법 대비 McNemar 정확 + Holm(본문 비교군 묶음).
실행: python3 results/mh_only_20260928/final/extract.py
"""
import hashlib
import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
RET = "results/mh_only_20260928/retrieval"
GROUPS = ["lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+"]

# (이름, 검색 기록, 답변 파일, 본문 여부)
ARMS = [
    ("본 방법 (셀 문장)", f"{RET}/mh_train_s3c_records.jsonl", "results/mh_arms/cap300_20260924/cell_uniq.jsonl", True),
    ("TableRAG(Yu) 청크", f"{RET}/mh_train_trag_hetero_records.jsonl", "results/mh_arms/cap300_20260924/trag_hetero.jsonl", True),
    ("TableRAG(Chen) leaf", f"{RET}/mh_train_tablerag_leaf_records.jsonl", "results/tablerag_official_20260928/answers/mh_tablerag_leaf_official.jsonl", True),
    ("TableRAG(Chen) path", f"{RET}/mh_train_tablerag_path_records.jsonl", "results/tablerag_official_20260928/answers/mh_tablerag_path_official.jsonl", True),
    ("참고: 표 전체 입력", None, "results/mh_arms/cap300_20260924/fulltable.jsonl", True),
    ("부록: 고정 청크", f"{RET}/mh_train_chunk_records.jsonl", "results/reader_format_20260927/test/mh_chunk_final.jsonl", False),
]
READER = {"prompt": "cot", "max_new_tokens": 384, "batch_size": 128}


def jl(rel):
    return [json.loads(x) for x in open(ROOT / rel, encoding="utf-8") if x.strip()]


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def mc(ref, x, ids):
    b = sum(1 for q in ids if ref[q] and not x[q])
    c = sum(1 for q in ids if x[q] and not ref[q])
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


manifest, ret, ans, layer = {}, {}, {}, {}
for name, rrel, arel, _main in ARMS:
    if rrel:
        meta = json.loads((ROOT / rrel.replace("_records.jsonl", ".json")).read_text())
        assert (meta["split"], meta["header_rule"], meta["alpha"], meta["budget_cells"]) == ("train", "v3.3u", 0.7, 20), (name, meta)
        assert meta["label_mix"] is None, name
        recs = {r["query_id"]: r for r in jl(rrel) if "doc" in r}
        assert len(recs) == 2885, (name, len(recs))
        ret[name] = {q: int(r["doc"]["correct"]) for q, r in recs.items()}
        layer.update({q: r["layer"] for q, r in recs.items()})
    run = json.loads((ROOT / arel.replace(".jsonl", ".run.json")).read_text())
    assert all(run[k] == v for k, v in READER.items()), (name, {k: run[k] for k in READER})
    assert run["reader"] == json.loads((ROOT / ARMS[0][2].replace(".jsonl", ".run.json")).read_text())["reader"], name
    rows = {r["query_id"]: r for r in jl(arel)}
    assert len(rows) == 882, (name, len(rows))
    if rrel:  # 리더가 받은 문맥 = 오늘 검색 기록의 문맥 (문항마다)
        bad = [q for q, r in rows.items() if r["source_context_sha256"] != recs[q]["doc"]["context_sha256"]]
    else:
        bad = []
    if not rrel:
        assert run["condition"] == "fulltable", name
    ans[name] = {q: int(r["answer_correct"]) for q, r in rows.items()}
    manifest[name] = {"retrieval": rrel, "retrieval_sha256": rrel and sha(rrel), "answer": arel, "answer_sha256": sha(arel),
                      "header_rule_answer_run": run["header_rule"], "condition": run["condition"],
                      "context_mismatch_vs_retrieval": len(bad), "context_mismatch_ids": bad}

ids882 = sorted(ans[ARMS[0][0]])
assert all(sorted(v) == ids882 for v in ans.values())
ids = sorted(ret[ARMS[0][0]])
assert all(sorted(v) == ids for v in ret.values())

base = ARMS[0][0]
res = {"retrieval_2885": {}, "answer_882": {}}
for name, rrel, _a, main in ARMS:
    if rrel:
        g = {k: round(sum(ret[name][q] for q in ids if layer[q] == k) / sum(1 for q in ids if layer[q] == k), 4) for k in GROUPS}
        b, c, p = mc(ret[base], ret[name], ids)
        res["retrieval_2885"][name] = {"correct": sum(ret[name].values()), "rate": round(sum(ret[name].values()) / 2885, 4),
                                       "by_group": g, "b": b, "c": c, "p": p}
    b, c, p = mc(ans[base], ans[name], ids882)
    res["answer_882"][name] = {"correct": sum(ans[name].values()), "rate": round(sum(ans[name].values()) / 882, 4), "b": b, "c": c, "p": p, "main": main}

for key in res:  # Holm: 본문 비교군만 한 묶음
    fam = sorted((n for n in res[key] if n != base and dict((a[0], a[3]) for a in ARMS)[n]), key=lambda n: res[key][n]["p"])
    run_max = 0.0
    for i, n in enumerate(fam):
        run_max = max(run_max, min(1.0, (len(fam) - i) * res[key][n]["p"]))
        res[key][n]["p_holm"] = run_max

(OUT / "final.json").write_text(json.dumps({"manifest": manifest, **res}, ensure_ascii=False, indent=1))
lines = ["# 줄인 비교군 결과 (2026-09-29, extract.py)", "",
         "## 검색 정확도 (2,885, 문서 안, 전달 셀 20, 정답 셀 전부 포함)", "",
         "| 방법 | 맞힘 | 비율 | " + " | ".join(GROUPS) + " | 본 방법만:상대만 | p | Holm p |", "|---" * (6 + len(GROUPS)) + "|"]
for n, v in res["retrieval_2885"].items():
    lines.append(f"| {n} | {v['correct']} | {v['rate']:.2%} | " + " | ".join(f"{v['by_group'][k]:.2%}" for k in GROUPS)
                 + (" | — | — | — |" if n == base else f" | {v['b']}:{v['c']} | {v['p']:.2g} | {v.get('p_holm', float('nan')):.2g} |"))
lines += ["", "## 답변 정확도 (882, Qwen3-8B cot 384, EM)", "", "| 방법 | 맞힘 | 비율 | 본 방법만:상대만 | p | Holm p | 검색 기록과 문맥 불일치 |", "|---|---|---|---|---|---|---|"]
for n, v in res["answer_882"].items():
    mm = manifest[n]["context_mismatch_vs_retrieval"] if manifest[n]["retrieval"] else "—"
    lines.append(f"| {n} | {v['correct']} | {v['rate']:.2%}" + (" | — | — | —" if n == base else
                 f" | {v['b']}:{v['c']} | {v['p']:.2g} | " + (f"{v['p_holm']:.2g}" if "p_holm" in v else "—")) + f" | {mm} |")
(OUT / "final.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
