"""검색 지표 표 — 본 방법 + 비교군 4개, Recall@k·Precision@k(k=1,5,10,20)·MRR (rpm.py 정의). 새 실행 없음.

범위: 앞 200 문항(5개 방법 모두), 882·2,885(검색에 LLM 이 필요 없는 방법만: 본 방법·LangChain·Yu 1회차 목록).
단위 사실: 단위당 평균 셀 수, 순위 목록 평균 길이 — k 는 단위 개수라 방법마다 k 개가 담는 셀 수가 다르다.
실행: .venv/bin/python results/mh_only_20260928/final/retrieval_table.py   (저장소 rag-agent 에서)
출력: final/retrieval_table.json, final/retrieval_table.md
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rpm import per_query, units_of  # noqa: E402

B = "results/mh_only_20260928/baselines/"
METHODS = [("본 방법 (셀 문장)", "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"),
           ("LangChain 튜토리얼 청크", B + "langchain_chunk/langchain_chunk_records.jsonl"),
           ("RAPTOR", B + "raptor/raptor_records.jsonl"),
           ("TableRAG (Chen 2024)", B + "tablerag_chen/tablerag_chen_records.jsonl"),
           ("TableRAG (Yu 2025)", B + "tablerag_yu/tablerag_yu_records.jsonl")]
KEYS = ["R@1", "R@5", "R@10", "R@20", "P@1", "P@5", "P@10", "P@20", "MRR"]


def ids(p):
    return [json.loads(l)["query_id"] for l in open(p, encoding="utf-8") if l.strip()]


def main():
    scopes = {"200": set(ids(B + "ids200.jsonl")), "882": set(ids("results/mh_arms/cap300_20260924/cell_uniq.jsonl")),
              "2885": {r["query_id"] for r in map(json.loads, open(METHODS[0][1], encoding="utf-8"))
                       if r.get("gold_ids") and units_of(r) is not None}}  # 본 방법이 채점한 2,885 (제외 23 은 gold·단위 없음)
    res = {}
    for name, path in METHODS:
        per, cpu, lens, cq = {}, [], [], {}
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            u = units_of(r)
            if u is None or not r.get("gold_ids"):
                continue
            per[r["query_id"]] = per_query(set(r["gold_ids"]), u)
            cpu += [len(x) for x in u]
            cq[r["query_id"]] = [len(x) for x in u]
            lens.append(len(u))
        res[name] = {"records": path, "cells_per_unit": round(statistics.mean(cpu), 1), "list_len": round(statistics.mean(lens), 1)}
        for tag, keep in scopes.items():
            q = [k for k in per if k in keep]
            if len(q) == len(keep):
                res[name][tag] = {"n": len(q), **{k: round(statistics.mean(per[x][k] for x in q), 4) for k in KEYS},
                                  "cells_per_unit": round(statistics.mean(c for x in q for c in cq[x]), 1)}
    assert all("200" in v for v in res.values()), "200 문항이 빠진 방법이 있다"
    (HERE / "retrieval_table.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    L = []
    for tag in ("200", "882", "2885"):
        L += [f"## {tag} 문항", "", "| 방법 | " + " | ".join(KEYS) + " | 단위당 셀 | 목록 길이 |", "|---" * (len(KEYS) + 3) + "|"]
        for name, v in res.items():
            if tag in v:
                L.append(f"| {name} | " + " | ".join(f"{v[tag][k]:.3f}" for k in KEYS) + f" | {v['cells_per_unit']} | {v['list_len']} |")
        L.append("")
    (HERE / "retrieval_table.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
