# SPDX-License-Identifier: MIT
"""리더의 `gold`/`oracle` 조건이 색인과 같은 문장을 주는가.

2026-09-09: 색인(`build_corpus`)은 ToTTo 페이지 제목을 붙이는데
`gold_context` 는 안 붙였다. 그래서 리더 천장은 코퍼스에 없던 제목
('career statistics')으로 쟀고, 검색 조건은 있던 제목
('Hristo Yanev: career statistics')으로 쟀다 — 두 조건이 다른 렌더링
위에서 비교되고 있었다. 주지표 검색 성공 906건 중 188건.

제목 내용이 EM 을 움직인다는 것은 이미 측정돼 있다(CLAUDE.md §5 생성 제목
기각: .3082 -> .2239). 그러므로 이 어긋남은 표기 문제가 아니라 측정 결함이다.

불변식: gold 셀이 문맥에 있다고 채점된 질의라면, `gold_context` 가 만든 문장이
리더가 실제로 받은 문맥 줄에 **그대로** 있어야 한다.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hitab"
RECORDS = ROOT / "results/retrieval_accuracy/t_s3c_hybrid_records.jsonl"
sys.path.insert(0, str(ROOT))

needs = pytest.mark.skipif(not (DATA.exists() and RECORDS.exists()),
                           reason="data/hitab 또는 본 방법 records 없음")


@needs
def test_gold_sentence_is_verbatim_in_delivered_context():
    _s = importlib.util.spec_from_file_location("aa", ROOT / "scripts/answer_accuracy.py")
    aa = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(aa)

    tabs, checked, missing = {}, 0, []
    for line in RECORDS.open():
        r = json.loads(line)
        # m==1 이라야 gold 문장이 하나로 떨어져 대조가 모호하지 않다.
        if not (r.get("correct") and r.get("m") == 1 and r.get("context")):
            continue
        gold = aa.gold_context(r, tabs, str(DATA))
        if len(gold) != 1:
            continue
        checked += 1
        if gold[0] not in r["context"]:
            missing.append((r["query_id"], gold[0], r["context"][0]))

    assert checked > 500, f"대조 표본이 너무 작다 ({checked})"
    assert not missing, (
        f"{len(missing)}/{checked} 건에서 gold 문장이 문맥에 없다. "
        f"예: {missing[0][1]!r} vs {missing[0][2]!r}")
