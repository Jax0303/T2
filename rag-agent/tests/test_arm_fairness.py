# SPDX-License-Identifier: MIT
"""표 1 의 공정성 — arm 을 추가할 때 자동으로 걸리게 해 두는 검사.

2026-09-08 에 색인 단위를 넷 얹으면서 라벨 다섯 건과 구현 다섯 건이 틀렸다. 그중
마지막(청킹이 셀 2.9% 를 잃던 것)은 사람이 "마지막으로 한 번 더 확인하라"고 해서
나왔다. 사람의 마지막 질문에 기대는 대신 여기에 못을 박는다.

검사하는 것은 표 1 이 비교로 성립하기 위한 조건들이다:
  - 모든 arm 이 같은 질의·같은 gold·같은 제외를 채점했는가
  - covers 가 없는 셀을 주장하지 않는가 (점수 부풀림)
  - covers 가 셀을 잃지 않는가 (baseline handicap) — tablerag 만 설계상 예외
  - 채점 코드가 arm 마다 갈라지지 않는가
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hitab"
RES = ROOT / "results/retrieval_accuracy"
_s = importlib.util.spec_from_file_location("ra", ROOT / "scripts/retrieval_accuracy.py")
ra = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ra)

needs_data = pytest.mark.skipif(not DATA.exists(), reason="data/hitab 없음")

#: (unit, build_corpus kwargs, 셀을 전부 배달해야 하는가)
UNITS = [("cell", {}, True), ("row", {"row_text": "values"}, True),
         ("rowcol", {"row_text": "values"}, True), ("chunk", {}, True),
         ("trag_hetero", {}, True), ("table", {}, True),
         ("tablerag", {}, False)]     # 숫자 열을 min/max 로 접으므로 부분집합


@needs_data
@pytest.mark.parametrize("unit,kw,complete", UNITS)
def test_covers_are_valid_and_lossless(unit, kw, complete):
    from rag_agent.bench import hitab_grid as hg
    tids = hg.table_ids(str(DATA))[:25]
    live = set()
    for tid in tids:
        tb = hg.load_table(tid, str(DATA))
        if tb is None:
            continue
        t = tb.table
        live |= {(tid, i, j) for i in range(t.n_rows) for j in range(t.n_cols)
                 if str(t.data[i][j]).strip()}
    texts, covers, *_ = ra.build_corpus(str(DATA), tids, "s3c", unit, {}, **kw)
    assert len(texts) == len(covers)
    got = set().union(*covers) if covers else set()
    assert got <= live, f"{unit}: 없는 셀 {len(got - live)}개를 주장한다"
    if complete:
        assert got == live, f"{unit}: 셀 {len(live - got)}개를 잃는다 (baseline handicap)"


@needs_data
def test_scoring_has_no_per_arm_branch():
    src = (ROOT / "scripts/retrieval_accuracy.py").read_text()
    seg = src[src.index('gold = q["gold"]'):src.index("recs.append(r)")]
    assert "a.unit" not in seg and "unit ==" not in seg, "채점이 arm 마다 갈라진다"
    assert seg.count("hit =") == 1, "판정 식이 하나가 아니다"


@pytest.mark.skipif(not (RES / "t_s3c_hybrid_records.jsonl").exists(),
                    reason="결과 파일 없음")
def test_every_arm_scored_the_same_queries():
    def recs(p):
        return {j["query_id"]: j for j in map(json.loads, p.open())}
    ref = recs(RES / "t_s3c_hybrid_records.jsonl")
    others = sorted(RES.glob("t_*_records.jsonl"))
    assert len(others) > 1
    for f in others:
        if "answer" in f.name:
            continue
        o = recs(f)
        assert set(o) == set(ref), f"{f.name}: 질의 집합이 다르다"
        for q in ref:
            assert o[q].get("m") == ref[q].get("m"), f"{f.name}/{q}: gold 셀 수가 다르다"
            assert o[q].get("excluded") == ref[q].get("excluded"), \
                f"{f.name}/{q}: 제외 사유가 다르다"


def test_budget_counts_distinct_cells():
    """예산은 리더가 받는 **서로 다른** 셀 수다.

    겹치는 색인 단위(trag_hetero 는 200자 overlap)가 이미 배달된 셀을 다시 실어도
    예산을 쓰지 않아야 한다. 중복을 세면 그 arm 만 두 번 손해다 -- 리더가 얻지도
    않은 셀에 예산을 쓰고 멈추고, 보고되는 문맥 셀수는 프롬프트에 있지도 않은
    크기를 적는다. `rowcol_select` 와 같은 규칙이라야 두 열이 같은 것을 잰다.
    """
    covers = [frozenset({("t", 0, j) for j in range(10)}),      # 0-9
              frozenset({("t", 0, j) for j in range(5, 15)}),   # 5-14 (5 겹침)
              frozenset({("t", 0, j) for j in range(15, 25)})]  # 15-24
    texts = ["a", "b", "c"]
    got, n, _ctx = ra.budget_select(range(3), covers, texts, 20, 0)
    assert n == len(got), "보고된 문맥 셀수가 실제 셀 수와 다르다"
    assert n == 25, f"겹친 5셀만큼 일찍 멈췄다 (n={n})"


def test_budget_never_scores_a_cell_it_did_not_deliver():
    """채점한 집합 = 배달한 집합. `dump` 는 두 번째 예산이 아니다.

    셀을 배달하지 않는 문서(TableRAG 의 schema/stub 문서)가 gold 앞에 오면,
    예전 코드는 `dump` 를 넘어선 단위까지 채점하면서 문맥에는 싣지 않아
    **gold 가 없는 문맥이 HIT 로** 기록됐다. 검색 정확도만 오르고 리더는 그 셀을
    본 적이 없다.

    고치는 방향은 **배달을 채점에 맞추는 것**이지 그 반대가 아니다. 훑기를 `dump`
    에서 끊으면 운영점이 "20셀"에서 "20단위"로 옮겨 가는데, 그 차이는 셀 0개
    단위를 갖는 arm 에서만 생긴다 — 우리가 만들어 낸 handicap 이고 이 파일이
    막으려는 바로 그것이다. 단위를 제한하려면 `--max-units` 가 따로 있다.
    """
    covers = [frozenset(), frozenset(), frozenset({("t", 5, 5)})]
    texts = ["schema1", "schema2", "GOLD"]
    got, n, ctx = ra.budget_select(range(3), covers, texts, budget=20, dump=2)
    assert ctx == ["schema1", "schema2", "GOLD"], ctx
    assert got == {("t", 5, 5)} and n == 1, got

    # 단위 제한은 명시적인 손잡이로만. 그때도 채점과 배달은 같은 집합이다.
    got, n, ctx = ra.budget_select(range(3), covers, texts, budget=20, dump=2,
                                   max_units=2)
    assert ctx == ["schema1", "schema2"] and got == set() and n == 0, (ctx, got)
