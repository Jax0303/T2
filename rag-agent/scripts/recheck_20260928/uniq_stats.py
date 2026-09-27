"""논문 3.3절(03_method.md 39·47줄)·6.5절(06_discussion.md 97줄)의 셀 문장 중복·고유화 수치 재계산 (2026-09-28).

실행:  cd rag-agent && HF_HUB_OFFLINE=1 .venv/bin/python scripts/recheck_20260928/uniq_stats.py
출력:  results/recheck_20260928/uniq_stats.json
임베딩·검색·생성 없음. 색인 코드(scripts/mh_arms.py build_tables, 셀 문장 렌더)로 셀 문장만 만들어 센다.

계산 전에 고정한 조건 (원고 문장 기준)
- 데이터: bevaya/MultiHiertt (캐시 snapshot ae21d2d, 검색 실행과 같음), 표 제목 없음(label none), 템플릿 s3c.
- 데이터 셀: 색인의 셀 단위(비어 있지 않은 데이터 칸) — build_corpus 와 같은 정의.
- "값을 뺀 문장": 색인 문장 렌더러에 값 없이(value=None) 넣은 문장 = 머리글 경로 문자열.
- "같은 문서": 같은 uid.
- train 범위: 검색 평가 색인(표 근거만 필요한 2,908 질의의 문서, keep_hybrid 없음). 원고 39줄이 분모를
  429,048(v1)·423,473(v3.3)으로 적어 이 색인으로 정해진다.
- 47줄 "train 분할에서 문장이 바뀐 셀"은 분모가 없어 문장만으로 범위가 정해지지 않는다(train 문서 전체 / 표 근거만 /
  표 근거+혼합). 06_discussion.md 97줄은 분할도 적지 않는다("MultiHiertt에서"). 처음에는 47줄을 39줄과 같은 색인으로
  고정했으나(첫 실행), 규칙에 따라 범위 해석 여섯 개(train·validation × 셋)를 모두 세고 어느 것도 출처로 연결하지 않는다.
- 39줄 비율 = 같은 문서 안에 값을 뺀 문장이 같은 다른 셀이 있는 셀 수 / 셀 수. 규칙 v1, v3.3.
- 47줄 "문장이 바뀐 셀" = v3.3 대 v3.3u 에서 값을 뺀 문장이 다른 셀. 세 종류(겹침 허용):
    표 안의 글 = 단계 1이 더한 경로 요소, 행·열 번호 = 경로 끝의 'row N'/'column N'(단계 2),
    표 번호 = 행 경로 앞의 'Table k'(단계 3).
  분모는 전체 셀(423,473). "그중 … 2.7%"를 바뀐 셀 대비로 읽으면 세 비율의 합이 100% 이상이어야 하는데
  (바뀐 셀은 적어도 하나가 더해짐) 원고 합은 15.8%라 성립하지 않으므로 전체 셀 대비 하나로 고정했다.
  바뀐 셀 대비 값은 참고로만 함께 적는다.
- 47줄 "적용 후 같은 문장인 셀 0개(train·validation)": validation 범위가 원고에 정해져 있지 않다
  (dev 911·929·332 질의 색인이 모두 쓰였다). 해석 셋을 모두 센다: 분할 전체 문서, 표 근거만(keep_hybrid 없음),
  표 근거+혼합(keep_hybrid). train 도 같은 셋을 센다. 문장은 값을 뺀 문장과 값까지 넣은 문장 둘 다 센다.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from datasets import load_dataset                                    # noqa: E402
from mh_arms import build_tables, load_population                    # noqa: E402
from retrieval_accuracy import cell_unit                             # noqa: E402
from rag_agent.serialization.caption import caption_sentence         # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT     # noqa: E402

OUT = ROOT / "results/recheck_20260928/uniq_stats.json"


def cells(tables):
    """{(tid, i, j): (row_path, col_path, value)} — 색인 셀 단위와 같은 집합."""
    out = {}
    for tid, tab in tables.items():
        t = tab.table
        for i in range(t.n_rows):
            for j in range(t.n_cols):
                if str(t.data[i][j]).strip():
                    out[tid, i, j] = (list(t.row_path(i)), list(t.col_path(j)), t.data[i][j])
    return out


def novalue(rp, cp):
    return caption_sentence("", rp, cp, value=None, template=STRUCTURAL_COMPACT)


def dup_cells(cs, with_value=False):
    """같은 문서 안에 같은 문장인 다른 셀이 있는 셀 수."""
    key = (lambda k, v: (k[0].split("::")[0], cell_unit("", v[0], v[1], v[2], "s3c"))) if with_value else \
          (lambda k, v: (k[0].split("::")[0], novalue(v[0], v[1])))
    n = Counter(key(k, v) for k, v in cs.items())
    return sum(1 for k, v in cs.items() if n[key(k, v)] > 1)


def build_u(docs):
    """v3.3u 표. 코드가 중복 잔존으로 멈추는 문서는 따로 센다."""
    try:
        return build_tables(docs, "v3.3u")[0], []
    except ValueError:
        tables, failed = {}, []
        for uid in sorted(docs):
            try:
                tables.update(build_tables({uid: docs[uid]}, "v3.3u")[0])
            except ValueError:
                failed.append(uid)
        return tables, failed


def all_docs(split):
    return {r["uid"]: (r["tables"], r["table_description"], r["paragraphs"])
            for r in load_dataset("bevaya/MultiHiertt", split=split)}


def change_stats(c3, cu):
    kind = Counter()
    for k, (rp3, cp3, _) in c3.items():
        rpu, cpu, _ = cu[k]
        kind.update(change_kind(k, rp3, cp3, rpu, cpu))
    n = len(cu)
    return {"n_cells": n, "changed": kind["changed"], "changed_rate": kind["changed"] / n,
            **{f"{a}_rate_of_all_cells": kind[a] / n for a in ("table_text", "row_col_number", "table_number")},
            **{a: kind[a] for a in ("table_text", "row_col_number", "table_number")},
            "참고_바뀐_셀_대비": {a: kind[a] / kind["changed"] for a in ("table_text", "row_col_number", "table_number")}}


def change_kind(k, rp3, cp3, rpu, cpu):
    if novalue(rp3, cp3) == novalue(rpu, cpu):
        return set()
    tbl = f"Table {int(k[0].split('::')[1]) + 1}"
    add = {"changed"}
    if rpu[:1] == [tbl] and rp3[:1] != [tbl]:
        add.add("table_number")
        rpu = rpu[1:]
    if rpu[-1:] == [f"row {k[1] + 1}"] and rp3[-1:] != rpu[-1:]:
        add.add("row_col_number")
        rpu = rpu[:-1]
    if cpu[-1:] == [f"column {k[2] + 1}"] and cp3[-1:] != cpu[-1:]:
        add.add("row_col_number")
        cpu = cpu[:-1]
    if rpu != rp3 or cpu != cp3:
        add.add("table_text")
    return add


out = {"conditions": __doc__.split("계산 전에 고정한 조건 (원고 문장 기준)")[1].strip()}

# ---- 39줄: v1·v3.3 중복 비율, 47줄: v3.3 → v3.3u 변화 (train 검색 평가 색인)
_, docs_tr, _ = load_population("train")
c1 = cells(build_tables(docs_tr, "v1")[0])
c3 = cells(build_tables(docs_tr, "v3.3")[0])
tu, failed = build_u(docs_tr)
cu = cells(tu)
assert not failed and cu.keys() == c3.keys()
out["train_index"] = {"n_docs": len(docs_tr)}
for name, cs in (("v1", c1), ("v3.3", c3)):
    d = dup_cells(cs)
    out["train_index"][name] = {"n_cells": len(cs), "dup_cells": d, "dup_rate": d / len(cs)}

out["train_index"]["v3.3_to_v3.3u"] = change_stats(c3, cu)

# ---- 47줄: 적용 후 같은 문장인 셀 (train·validation, 범위 해석 셋)
out["residual_after_v3.3u"], out["change_by_scope"] = {}, {}
for split in ("train", "validation"):
    scopes = {"split_all_docs": all_docs(split),
              "table_evidence_only": load_population(split)[1],
              "table_evidence_plus_hybrid": load_population(split, keep_hybrid=True)[1]}
    for scope, docs in scopes.items():
        tables, failed = (tu, []) if (split, scope) == ("train", "table_evidence_only") else build_u(docs)
        cs = cells(tables)
        out["residual_after_v3.3u"][f"{split}.{scope}"] = {
            "n_docs": len(docs), "n_cells": len(cs), "docs_stopped_by_code_check": len(failed),
            "dup_cells_novalue": dup_cells(cs), "dup_cells_with_value": dup_cells(cs, True)}
        c3s = c3 if (split, scope) == ("train", "table_evidence_only") else cells(build_tables(docs, "v3.3")[0])
        assert not failed and c3s.keys() == cs.keys()
        out["change_by_scope"][f"{split}.{scope}"] = change_stats(c3s, cs)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "conditions"}, ensure_ascii=False, indent=2))
