# SPDX-License-Identifier: MIT
"""MultiHiertt 답변 레그의 문항별 즉시 저장과 재개.

중단 후 재개해도 누락·중복이 없고, 문항 id·문맥 해시·모델 revision·프롬프트·생성 설정 중 하나라도
저장된 실행과 다르면 이어 쓰지 않는다.
"""
import json
import sys
from types import SimpleNamespace

import pytest

from rag_agent.eval.artifacts import digest, file_digest, read_records, write_pair
from scripts import answer_accuracy_mh as am

IDS = [f"q{i}" for i in range(5)]


def fixture(tmp_path):
    rows = []
    for q in IDS:
        ctx = [f"row > col {q}: 7"]
        rows.append({"query_id": q, "layer": "lookup_m1", "kind": "lookup", "m": 1,
                     "question": f"value of {q}?", "answer": "7",
                     "doc": {"correct": 1, "cells_in_context": 1, "context": ctx,
                             "context_sha256": digest(ctx)}})
    rp = tmp_path / "fx_records.jsonl"
    write_pair(rp, rows, {"context_version": 2}, summary_path=tmp_path / "fx.json")
    return rp


class Reader:
    name = "fixture"
    context_limit = 1000

    def __init__(self, revision="r1", fail_after=None):
        self.revision, self.fail_after, self.calls = revision, fail_after, []

    def metadata(self):
        return {"name": self.name, "revision_resolved": self.revision, "chat_template": "t"}

    def n_prompt_tokens(self, system, user):
        return 20

    def complete(self, system, user, **kwargs):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("simulated crash")
        self.calls.append(user)
        return "7"


def run(monkeypatch, rp, out, reader, *extra):
    monkeypatch.setattr(am, "build_llm", lambda spec: reader)
    monkeypatch.setattr(am, "official_answers", lambda split: {q: "7" for q in IDS})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    # 실제 torch 는 상대 경로 __file__('_ops.py')로 모듈을 등록한다 — 실행 조건 해시가 여기서 멈춘 전례
    monkeypatch.setitem(sys.modules, "_relative_file_module", SimpleNamespace(__file__="_ops.py"))
    monkeypatch.setattr(sys, "argv", ["answer_accuracy_mh.py", "--records", str(rp),
                                      "--out", str(out), *extra])
    return am.main()


def crashed(tmp_path, monkeypatch):
    rp, out = fixture(tmp_path), tmp_path / "ans.jsonl"
    with pytest.raises(RuntimeError):
        run(monkeypatch, rp, out, Reader(fail_after=2))
    return rp, out


def test_crash_then_resume_has_no_gap_or_duplicate(tmp_path, monkeypatch):
    rp, out = crashed(tmp_path, monkeypatch)
    assert [json.loads(l)["query_id"] for l in out.open()] == IDS[:2]   # 끝난 문항은 이미 디스크에
    assert not out.with_suffix(".json").exists()
    with out.open("a") as f:
        f.write('{"query_id": "q2", "lay')                              # 쓰다 끊긴 줄
    with pytest.raises(SystemExit, match="--resume"):
        run(monkeypatch, rp, out, Reader())

    reader = Reader()
    assert run(monkeypatch, rp, out, reader, "--resume") == 0
    assert len(reader.calls) == 3                                       # 남은 문항만 생성
    assert [json.loads(l)["query_id"] for l in out.open()] == IDS       # 누락·중복 없음, 순서 유지
    assert set(read_records(out)) == set(IDS)
    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["records_sha256"] == file_digest(out)
    assert summary["resumed_rows"] == 2 and summary["by_layer"]["ALL"]["n"] == 5
    with pytest.raises(SystemExit, match="exists"):
        run(monkeypatch, rp, out, Reader(), "--resume")                 # 완료된 레그는 다시 돌지 않는다


@pytest.mark.parametrize("change", ["revision", "prompt", "max_tokens"])
def test_resume_refuses_different_reader_setting(tmp_path, monkeypatch, change):
    rp, out = crashed(tmp_path, monkeypatch)
    before = out.read_bytes()
    reader, extra = Reader(), ["--resume"]
    if change == "revision":
        reader = Reader(revision="r2")
    elif change == "prompt":
        extra += ["--prompt", "cot"]
    else:
        extra += ["--max-tokens", "32"]
    with pytest.raises(SystemExit, match="재개 조건"):
        run(monkeypatch, rp, out, reader, *extra)
    assert out.read_bytes() == before and not reader.calls              # 거절하면 생성도 쓰기도 없다


@pytest.mark.parametrize("tamper, message", [("context", "문맥"), ("duplicate", "두 번"), ("gap", "누락")])
def test_resume_refuses_inconsistent_saved_rows(tmp_path, monkeypatch, tamper, message):
    rp, out = crashed(tmp_path, monkeypatch)
    rows = [json.loads(l) for l in out.open()]
    if tamper == "context":
        rows[1]["context_sha256"] = "other"
    elif tamper == "duplicate":
        rows.append(rows[0])
    else:
        rows = rows[1:]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    reader = Reader()
    with pytest.raises(SystemExit, match=message):
        run(monkeypatch, rp, out, reader, "--resume")
    assert not reader.calls
