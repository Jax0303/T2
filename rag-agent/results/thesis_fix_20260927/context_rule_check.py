"""MultiHiertt 색인·답변 문맥이 머리글 규칙에 따라 바뀌는지 확인한다(2026-09-27). 검색·생성 실행은 없고 문장만 다시 만든다.

실행:  cd rag-agent && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/thesis_fix_20260927/context_rule_check.py
출력:  context_rule_check.json (같은 폴더)

1. 셀 문장(s3c, 라벨 없음 = 경로만): 지금 코드로 처음 규칙(v1)·최종 규칙(v3.3u) 문장을 다시 만들어 각 결과 파일의
   corpus_text_sha256 과 대조한다. 둘 다 같으면 두 결과는 같은 문장 규칙이고 입력 경로(머리글 규칙)만 다르다.
2. 비교군 색인 텍스트: 지금 코드로 v1 을 다시 만들어 v1 결과 파일 해시와 같은지(= 코드 변경 없음), v1 과 v3.3u 가 같은지.
3. 표 전체: 882건 문서의 markdown(answer_accuracy_mh.full_tables 와 같은 markdown_source)이 두 규칙에서 같은지.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import mh_arms as m                                          # noqa: E402
from retrieval_accuracy import build_corpus                  # noqa: E402
from rag_agent.eval.artifacts import digest                  # noqa: E402
from rag_agent.serialization.chunks import markdown_source   # noqa: E402

MA = ROOT / "results/mh_arms"
queries, docs, _ = m.load_population("train")
T = {r: m.build_tables(docs, r, "none")[0] for r in ("v1", "v3.3u")}
# 이름 -> (build_corpus 인자, v1 결과 파일 태그)
UNITS = {"cell_s3c": (dict(template="s3c", unit="cell"), "cell"),
         "chunk": (dict(unit="chunk"), "chunk"),
         "trag_hetero": (dict(unit="trag_hetero"), "trag_hetero"),
         "rowcol_values": (dict(unit="rowcol", row_text="values"), "rowcol_values"),
         "rowcol_sentence": (dict(unit="rowcol", row_text="sentence"), "rowcol"),
         "randrow_values": (dict(unit="row", row_text="values"), "randrow_values"),
         "randrow_sentence": (dict(unit="row", row_text="sentence"), "randrow"),
         "tablerag_path": (dict(unit="tablerag", colmode="path"), "tablerag_path"),
         "tablerag_leaf": (dict(unit="tablerag", colmode="leaf"), "tablerag_leaf")}
out = {"index_text": {}}
for name, (u, tag) in UNITS.items():
    sh = {}
    for r, tabs in T.items():
        texts, *_ = build_corpus("", sorted(tabs), u.get("template", "s3c"), u["unit"], {}, 1000, u.get("colmode", "leaf"),
                                 u.get("row_text", "values"), 200, None, load=lambda tid, _d, tabs=tabs: tabs.get(tid))
        sh[r] = digest(texts)
        if name == "cell_s3c":
            sh[f"{r}_label_frame_lines"] = sum("In the table" in t for t in texts)
    ref = json.loads((MA / f"mh_train_{tag}_hv1_none_doc.json").read_text())["corpus_text_sha256"]
    row = {"v1_rebuilt_equals_v1_result": sh["v1"] == ref, "v1_equals_v3.3u": sh["v1"] == sh["v3.3u"]}
    if name == "cell_s3c":
        fin = json.loads((ROOT / "results/rerun_20260926/mh/mh_train_s3c.json").read_text())["corpus_text_sha256"]
        row |= {"v3.3u_rebuilt_equals_rerun_s3c": sh["v3.3u"] == fin,
                "label_frame_lines": {r: sh[f"{r}_label_frame_lines"] for r in T}}
    out["index_text"][name] = row

ids = {q for g in json.loads((MA / "sample_cap300_seed20260913.json").read_text())["by_layer"].values() for q in g}
md = {}
for r, tabs in T.items():
    d = {}
    for t in sorted(tabs, key=lambda t: (t.split("::")[0], int(t.split("::")[1]))):
        if t.split("::")[0] in ids:
            d.setdefault(t.split("::")[0], []).append(markdown_source(tabs[t], tabs[t].table, tabs[t].title)[0])
    md[r] = d
out["fulltable_882"] = {"docs": len(ids), "docs_with_tables": {r: len(d) for r, d in md.items()},
                        "same_markdown_v1_v3.3u": sum(1 for u in ids if md["v1"].get(u) and md["v1"].get(u) == md["v3.3u"].get(u))}
Path(__file__).with_name("context_rule_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=1))
