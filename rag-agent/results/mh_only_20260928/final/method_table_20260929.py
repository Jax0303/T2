"""방법별 검색·답변 비교표 (2026-09-29). 새 실행 없음 — 저장된 파일만 읽어 표를 만든다.

검색 지표: rpm.py 의 units_of·per_query (k = 검색 단위 개수, 문항 평균). gold_ids·순위 단위가 없는 레코드는 제외하고 센다.
답변: 공식 EM = bool(answer_correct); DocMath 허용 = docmath_match(pred, answer, 본 방법 답변 파일의 kind == "arith").
McNemar 정확 검정·Holm 은 compare.py 의 mc·holm (본 방법 대비, b = 본 방법만 맞힘, c = 상대만 맞힘).
실행 중인 파일(Yu 답변, Chen 512) 은 200 문항이 다 차면 다음 실행에서 자동으로 계산된다.
실행: .venv/bin/python results/mh_only_20260928/final/method_table_20260929.py   (저장소 rag-agent 에서, 어디서든 가능)
출력: final/method_table_20260929.md
"""
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(HERE), str(ROOT)]
from rpm import per_query, units_of  # noqa: E402
from compare import holm, mc  # noqa: E402
from rag_agent.eval.multihiertt_em import docmath_match  # noqa: E402

B = "results/mh_only_20260928/baselines/"
OURS = "본 방법(셀 문장)"
ABL = "머리글 경로 제거(ablation)"
LC = "LangChain 청크"
CHEN = "TableRAG(Chen), 원 설정 128토큰"
YU = "TableRAG(Yu)"
FULL = "참고: 표 전체 입력"
RET = {OURS: "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl",
       ABL: "results/header_ablation_20260929/mh_train_s3c_value_records.jsonl",
       LC: B + "langchain_chunk/langchain_chunk_records.jsonl",
       "RAPTOR": B + "raptor/raptor_records.jsonl",
       CHEN: B + "tablerag_chen/tablerag_chen_records.jsonl",
       YU: B + "tablerag_yu/tablerag_yu_records.jsonl"}
ANS = {OURS: "results/mh_arms/cap300_20260924/cell_uniq.jsonl",
       ABL: "results/header_ablation_20260929/mh_value_answer.jsonl",
       LC: B + "langchain_chunk/langchain_chunk_answer.jsonl",
       "RAPTOR 2,000토큰": B + "raptor/raptor_answer_2000.jsonl",
       "RAPTOR 575토큰": B + "raptor/raptor_answer_575.jsonl",
       CHEN: B + "tablerag_chen/tablerag_chen_answer.jsonl",
       "TableRAG(Chen), 512토큰": B + "tablerag_chen_mt512/tablerag_chen_answer.jsonl",
       YU: B + "tablerag_yu/tablerag_yu_answer.jsonl",
       FULL: "results/mh_arms/cap300_20260924/fulltable.jsonl"}
EXTERNAL = {LC, "RAPTOR 2,000토큰", "RAPTOR 575토큰", CHEN, "TableRAG(Chen), 512토큰", YU}  # Holm 가족 (완료된 것만)
KEYS = ["R@1", "R@5", "R@10", "R@20", "P@1", "P@5", "P@10", "P@20"]


def jl(p):  # 끝 줄바꿈이 없는 줄은 실행 중 쓰다 만 줄로 보고 읽지 않는다
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.endswith("\n") and x.strip()]


def retrieval(path):
    per, ncells, skipped = {}, {}, 0
    for r in jl(path):
        u = units_of(r)
        if u is None or not r.get("gold_ids"):
            skipped += 1
            continue
        per[r["query_id"]] = per_query(set(r["gold_ids"]), u)
        raw = r.get("ranked_units") or r["doc"]["ranked_units"]
        ncells[r["query_id"]] = [len(x["cells"]) if isinstance(x, dict) else len(x) for x in raw]
    return per, ncells, skipped


def ret_row(per, ncells, scope):
    q = [x for x in scope if x in per]
    m = {k: statistics.mean(per[x][k] for x in q) for k in KEYS + ["MRR"]}
    m["n"] = len(q)
    m["cells"] = statistics.mean(c for x in q for c in ncells[x])
    m["len"] = statistics.mean(len(ncells[x]) for x in q)
    return m


def answer_rows(names, scope, ours, kind):
    rows, fam = {}, {}
    for name in names:
        p = ANS[name]
        if not os.path.exists(p):
            rows[name] = "실행 대기"
            continue
        a = {r["query_id"]: r for r in jl(p)}
        have = sum(q in a for q in scope)
        if have < len(scope):
            rows[name] = f"실행 중 ({have}/{len(scope)})"
            continue
        em = {q: bool(a[q]["answer_correct"]) for q in scope}
        dm = sum(bool(docmath_match(a[q].get("pred"), a[q]["answer"], kind[q] == "arith")) for q in scope)
        rows[name] = {"k": sum(em.values()), "dm": dm, **(mc(ours, em, scope) if name != OURS else {})}
        if name in EXTERNAL:
            fam[name] = rows[name]
    holm(fam)
    return rows


def pct(x, n=1):
    return f"{100 * x / n:.1f}"


def pfmt(p):
    return f"{p:.3g}"


def ret_table(res, names, tag):
    L = [f"| 방법 | 문항 수 | {' | '.join(KEYS)} | MRR | 단위당 평균 셀 수 | 문항당 순위 목록 평균 길이 |",
         "|---" * (len(KEYS) + 5) + "|"]
    for n in names:
        m = res[n][tag]
        L.append(f"| {n} | {m['n']:,} | " + " | ".join(pct(m[k]) for k in KEYS)
                 + f" | {m['MRR']:.3f} | {m['cells']:.1f} | {m['len']:.1f} |")
    return L


def ans_table(rows, n):
    L = ["| 방법 | 맞힌 수 | 정확도(%) | DocMath(%) | b(본 방법만) | c(상대만) | McNemar p | Holm p |", "|---" * 8 + "|"]
    for name, v in rows.items():
        if isinstance(v, str):
            L.append(f"| {name} | {v} | – | – | – | – | – | – |")
        elif name == OURS:
            L.append(f"| {name} | {v['k']} | {pct(v['k'], n)} | {pct(v['dm'], n)} | – | – | – | – |")
        else:
            h = pfmt(v["p_holm"]) if "p_holm" in v else "가족 제외"
            L.append(f"| {name} | {v['k']} | {pct(v['k'], n)} | {pct(v['dm'], n)} | {v['b']} | {v['c']} | {pfmt(v['p'])} | {h} |")
    return L


def main():
    ids200 = [json.loads(x)["query_id"] for x in open(B + "ids200.jsonl", encoding="utf-8") if x.strip()]
    ans_ours = {r["query_id"]: r for r in jl(ANS[OURS])}
    ids882 = sorted(ans_ours)
    kind = {q: r["kind"] for q, r in ans_ours.items()}
    ours_em = {q: bool(r["answer_correct"]) for q, r in ans_ours.items()}

    res, skips = {}, {}
    for name, path in RET.items():
        per, ncells, skips[name] = retrieval(path)
        if name == OURS:
            ids2885 = sorted(per)
        res[name] = {"per": per, "nc": ncells}
    for name, v in res.items():
        for tag, scope in (("200", ids200), ("882", ids882), ("2885", ids2885)):
            if tag == "200" or name in (OURS, ABL, LC, YU):
                v[tag] = ret_row(v["per"], v["nc"], scope)

    a200 = answer_rows([OURS, ABL, LC, "RAPTOR 2,000토큰", "RAPTOR 575토큰", CHEN, "TableRAG(Chen), 512토큰", YU, FULL],
                       ids200, ours_em, kind)
    a882 = answer_rows([OURS, LC, ABL, FULL], ids882, ours_em, kind)

    # 알려진 값 점검 — 하나라도 틀리면 파일을 쓰지 않는다
    for rows, want in ((a200, {OURS: 93, LC: 60, "RAPTOR 2,000토큰": 57, "RAPTOR 575토큰": 51, CHEN: 12, FULL: 80}),
                       (a882, {OURS: 408, LC: 298, ABL: 143, FULL: 378})):
        for name, k in want.items():
            assert rows[name]["k"] == k, (name, rows[name], k)
    o = res[OURS]["200"]
    assert (round(o["R@1"], 3), round(o["R@20"], 3), round(o["MRR"], 3), o["n"]) == (0.323, 0.939, 0.703, 200), o
    assert len(ids2885) == 2885 and len(ids882) == 882 and len(ids200) == 200

    ret_all = list(RET)
    L = ["# MultiHiertt 방법별 검색·답변 비교 (2026-09-29)", "",
         f"`{Path(__file__).relative_to(ROOT)}` 가 저장된 검색·답변 파일에서 계산한 표. 답변 생성 모델 Qwen3-8B.", "",
         "## 1. 검색 성능, 200문항", "", "R@k·P@k 는 %, k = 검색 단위 개수.", ""]
    L += ret_table(res, ret_all, "200")
    for tag, label in (("882", "882문항"), ("2885", "2,885문항")):
        L += ["", f"## 2. 검색 성능, {label}", ""] + ret_table(res, [n for n in ret_all if tag in res[n]], tag)
    L += ["", "## 3. 답변 정확도, 200문항", ""] + ans_table(a200, 200)
    L += ["", "## 4. 답변 정확도, 882문항", ""] + ans_table(a882, 882)
    fam200 = [n for n, v in a200.items() if isinstance(v, dict) and "p_holm" in v]
    L += ["", "## 주의", "",
          "- k 는 검색 단위(본 방법·머리글 제거는 셀 1개, 비교군은 청크·노드·셀 검색 문서) 개수다. 청크 하나에 셀이 여러 개 들어 있으므로 "
          "R@k 를 셀 단위 k 와 바로 비교할 수 없다 — '단위당 평균 셀 수' 열과 함께 읽는다.",
          "- 단위당 평균 셀 수·순위 목록 평균 길이는 그 표의 문항 집합에 속한 레코드의 순위 단위 전부로 계산했다.",
          "- RAPTOR·TableRAG(Chen) 은 검색 단계에 LLM 호출이 들어가 200문항만 실행했다.",
          "- TableRAG(Yu) 검색 순위 = 질문 자체로 한 1회차 검색(재정렬 상위 3 다음 나머지 회수 청크). LLM 질의 분해는 들어가지 않는다.",
          "- RAPTOR 요약 노드의 셀 = 자식 노드가 담은 셀의 합집합. RAPTOR 두 행(2,000토큰·575토큰)은 같은 검색 결과에서 문맥 길이 상한만 다르다.",
          "- 검색 표에서 제외한 레코드(gold 셀·순위 단위 없음), 파일 전체 기준: "
          + " / ".join(f"{n} {s}건" for n, s in skips.items()) + ".",
          "- 정확도 = 공식 MultiHiertt EM. DocMath = 산술 문항(본 방법 답변 파일의 kind 기준)만 DocMath 허용 비교, 나머지 문항은 공식 EM 과 같다.",
          "- McNemar = 정확 이항 검정(양측). Holm 가족 = 완료된 외부 비교군 "
          + (" / ".join(fam200) if fam200 else "없음") + " (200문항), LangChain 청크 1개 (882문항, 보정 전과 같음). "
          "머리글 경로 제거·표 전체 입력은 가족에서 빼고 보정 전 p 만 적었다.",
          "- '실행 대기' = 결과 파일 없음, '실행 중 (n/200)' = 200문항 중 n 문항만 기록됨 — 정확도를 계산하지 않았다."]
    out = HERE / "method_table_20260929.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
