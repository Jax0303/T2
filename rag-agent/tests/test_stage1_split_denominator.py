# SPDX-License-Identifier: MIT
"""스플릿 분모 절: 모집단이 겹쳐도 질의를 두 번 세지 않는다.

`hitab_dev_lookup_all` 과 `hitab_dev_corpus_arith` 는 실제로 35건 겹친다 (m=1 인 산술).
행 단위 n 을 더하면 1038, 실제 채점된 질의는 1003 이다. 이 절은 union 을 세야 한다.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rec(qid, rank):
    return {"query_id": qid, "m": 1, "gold_table": "t", "table_rank_cellvote": 1,
            "ranks": [rank], "ranks_in_table": [0]}


def test_union_not_sum(tmp_path):
    a = tmp_path / "hitab_dev_lookup_all_S3c_page_x_ranks.jsonl"
    b = tmp_path / "hitab_dev_corpus_arith_S3c_page_x_ranks.jsonl"
    a.write_text("\n".join(json.dumps(rec(q, 0)) for q in ("q1", "q2", "shared")) + "\n")
    b.write_text("\n".join(json.dumps(rec(q, 99)) for q in ("shared", "q3")) + "\n")
    out = tmp_path / "BOARD.md"
    subprocess.run([sys.executable, "analysis/stage1_board.py", str(a), str(b), "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True,
                   env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    line = [l for l in out.read_text().splitlines() if l.startswith("| `dev`")][0]
    scored = int(line.split("|")[3])
    assert scored == 4, line          # 5 가 아니라 4 — shared 는 한 번
    # 겹친 질의는 나중 파일이 이긴다: q1,q2 성공 / shared,q3 실패
    assert int(line.split("|")[5]) == 2, line
