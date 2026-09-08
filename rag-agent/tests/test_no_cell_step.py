# SPDX-License-Identifier: MIT
"""`NO_CELL_STEP` 은 손으로 고른 목록이 아니라 records 가 정하는 사실이어야 한다.

표 1 은 셀 검색 정확도를 잰다. 색인 단위가 표 하나면 gold 표가 뽑히는 순간 그 표의
셀이 전부 문맥에 들어오므로 셀을 고르는 단계가 없고, "표는 찾았는데 셀은 놓쳤다"가
한 건도 나올 수 없다. 그 행의 값을 다른 행과 같은 칸에 넣으면 두 열이 다른 것을
재게 되므로 비운다 — 그 판단의 근거가 이 불변식이다.

새 arm 을 추가했는데 그것도 축소되면(또는 반대로 목록에 있는 arm 이 더 이상
축소되지 않으면) 여기서 깨진다.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("at", ROOT / "analysis/accuracy_tables.py")
at = importlib.util.module_from_spec(_s)
_s.loader.exec_module(at)


def missed_cell_after_finding_table(tag):
    """all 모드에서 gold 표는 문맥에 있는데 검색은 틀린 질의 수, 그리고 채점 수."""
    p = ROOT / f"results/retrieval_accuracy/{tag}_records.jsonl"
    if not p.exists():
        return None
    n = bad = 0
    for line in p.open():
        r = json.loads(line)
        if r.get("mode") != "all" or "correct" not in r:
            continue
        n += 1
        bad += r["gold_table_in_context"] == 1 and r["correct"] == 0
    return bad, n


def test_no_cell_step_is_exactly_the_degenerate_arms():
    seen = {}
    for tag, _label, _ref, _kind in at.BASELINE_ROWS:
        got = missed_cell_after_finding_table(tag)
        if got is None:            # 아직 안 돈 arm 은 판정하지 않는다
            continue
        bad, n = got
        assert n > 0, f"{tag}: all 모드 채점이 0건"
        seen[tag] = bad
    assert seen, "records 가 하나도 없다 — 검색 레그를 먼저 돌려야 한다"
    degenerate = {t for t, bad in seen.items() if bad == 0}
    listed = at.NO_CELL_STEP & set(seen)
    assert degenerate == listed, (
        f"셀 검색 단계가 없는 arm 과 NO_CELL_STEP 이 어긋난다: "
        f"측정={sorted(degenerate)} 목록={sorted(listed)} (건수 {seen})")
