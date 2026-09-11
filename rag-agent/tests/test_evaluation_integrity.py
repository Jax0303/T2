# SPDX-License-Identifier: MIT
"""Regressions for the September 2026 evidence/EM audit; no model downloads."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_agent.eval.artifacts import (Selection, digest, evidence_fields, read_records,
                                     require_same_ids, validate_retrieval, write_pair)
from rag_agent.retrieve.encoders import SentenceTransformerEncoder
from rag_agent.serialization.chunks import split_chunks, markdown_source, table_chunks
from scripts import retrieval_accuracy as ra
from scripts import answer_accuracy as aa
from analysis.validated_tables import build, load_manifest, paired_counts, validate_answers
from analysis.compose_oracle import compose


def record(qid="q", n_gold=1):
    cells = {("T", 0, j) for j in range(n_gold)}
    selected = Selection(cells, [{"text": "value: 7", "cells": sorted(cells)}], True)
    return {"query_id": qid, "table_id": "T", "question": "value?", "answer": [7],
            "mode": "all", "m": n_gold, "correct": 1, "aggregation": None,
            "cells_in_context": n_gold, "gold_table_in_context": 1,
            **evidence_fields(selected, cells)}


def answer(r):
    return {"query_id": r["query_id"], "question": r["question"], "answer": r["answer"], "pred": "7",
            "mode": r["mode"], "aggregation": None, "retrieval_correct": r["correct"],
            "answer_correct": 1, "cells_in_context": r["cells_in_context"],
            "n_ctx": 1, "n_tok": 10, "context_sha256": r["context_sha256"],
            "source_context_sha256": r["context_sha256"]}


def test_export_limit_never_hides_selected_units():
    covers = [set(), set(), {("T", 0, 0)}, {("T", 0, 1)}]
    selection = ra.budget_select(range(4), covers, ["schema1", "schema2", "7", "8"], 2, dump=1)
    assert selection.context == ["schema1", "schema2", "7", "8"]
    assert len(selection.cells) == 2
    capped = ra.budget_select(range(4), covers, ["a", "b", "7", "8"], 2, dump=1, max_units=3)
    assert capped.context == ["a", "b", "7"] and len(capped.cells) == 1


def test_gold_above_64_is_saved_and_validated_in_full():
    row = record(n_gold=408)
    validate_retrieval(row)
    row["gold_cells"] = row["gold_cells"][:64]
    with pytest.raises(ValueError, match="truncated"):
        validate_retrieval(row)


@pytest.mark.parametrize("field,value", [
    ("context", ["other"]), ("correct", 0), ("cells_in_context", 99),
    ("context_cells", []), ("context_version", 1), ("context_sha256", "wrong")])
def test_evidence_mismatch_is_rejected(field, value):
    row = record()
    row[field] = value
    with pytest.raises(ValueError):
        validate_retrieval(row)


def test_rowcol_delivers_only_intersection_values_with_labels():
    class Table:
        data = [["SELECTED", "LEAK_ROW"], ["LEAK_COLUMN", "LEAK_OTHER"]]
        def col_path(self, j): return ["top", f"c{j}"]
        def row_path(self, i): return ["side", f"r{i}"]
    tab = SimpleNamespace(title="title", table=Table())
    covers = [{("T", 0, 0), ("T", 0, 1)}, {("T", 0, 0), ("T", 1, 0)}]
    selected = ra.rowcol_select([0, 1], covers, ["LEAK_ROW", "LEAK_COLUMN"],
                                [True, False], budget=1, dump=1,
                                render_context=lambda cs: ra.subtable_context(cs, {"T": tab}, "", {}))
    assert selected.cells == {("T", 0, 0)}
    text = "\n".join(selected.context)
    assert "SELECTED" in text and "side > r0" in text and "top > c0" in text
    assert "LEAK" not in text
    assert evidence_fields(selected, selected.cells)["context_cells"] == [("T", 0, 0)]


def test_rowcol_refuses_old_union_export_without_renderer():
    with pytest.raises(ValueError, match="renderer"):
        ra.rowcol_select([], [], [], [], 20, 1)


def test_character_chunks_equal_native_splitter_and_preserve_offsets():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text = "Table name: title\n" + "\n".join("| " + "abc def " * 80 + "|" for _ in range(3))
    native = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_text(text)
    chunks = list(split_chunks(text, 100, 20))
    assert [c[0] for c in chunks] == native
    assert all(text[start:end] == chunk and len(chunk) <= 100 for chunk, start, end in chunks)


def test_true_token_chunks_with_fast_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    text = "alpha beta gamma " * 30
    chunks = list(split_chunks(text, 10, 2, tokenizer))
    assert len(chunks) > 1
    assert all(len(tokenizer(chunk, add_special_tokens=False)["input_ids"]) <= 10
               and text[start:end] == chunk for chunk, start, end in chunks)
    assert chunks[-1][2] == len(text.rstrip())


def test_chunk_does_not_credit_a_partial_value():
    table = SimpleNamespace(data=[["abcdefghijklmno"]])
    tab = SimpleNamespace(raw={"texts": [["abcdefghijklmno"]]}, row_map={0: 0}, col_map={0: 0})
    assert all(not cells for _, cells in table_chunks(tab, table, "T", 5, 0))


def test_embedding_overflow_counts_prefix_and_special_tokens():
    def tokenize(batch, **kwargs):
        assert kwargs == {"truncation": False, "padding": False, "add_special_tokens": True}
        return {"input_ids": [[0] * (len(text.split()) + 2) for text in batch]}
    enc = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    enc.model = SimpleNamespace(tokenizer=tokenize, max_seq_length=5)
    enc.batch_size = 2
    enc.query_prefix, enc.passage_prefix = "query prefix ", ""
    assert enc.audit_inputs(["one two"]) ["n_overflow"] == 0
    with pytest.raises(ValueError, match="encoder input overflow"):
        enc.audit_inputs(["one two"], query=True)
    report = enc.audit_inputs(["one two"], query=True, overflow="truncate")
    assert report["max_tokens"] == 6 and report["overflow_indices"] == [0]


def test_duplicate_ids_and_subset_comparisons_fail(tmp_path):
    path = tmp_path / "dupes.jsonl"
    path.write_text(json.dumps(record()) + "\n" + json.dumps(record()) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_records(path)
    with pytest.raises(ValueError, match="IDs differ"):
        require_same_ids({"a", "b"}, {"a"})


def test_reader_accepts_only_matching_v2_summary(tmp_path):
    path = tmp_path / "run_records.jsonl"
    write_pair(path, [record()], {"context_version": 2}, summary_path=tmp_path / "run.json")
    rows, _ = aa.load_evidence(path)
    assert len(rows) == 1
    with path.open("a") as stream:
        stream.write(json.dumps(record("q2")) + "\n")
    with pytest.raises(ValueError, match="stale"):
        aa.load_evidence(path)


def test_writer_refuses_orphan_summary(tmp_path):
    path = tmp_path / "run.jsonl"
    path.with_suffix(".json").write_text("{}")
    with pytest.raises(FileExistsError):
        write_pair(path, [record()], {})
    assert not path.exists()


def test_reader_context_limit_includes_generation_budget():
    aa.check_context_limit(90, 10, 100)
    with pytest.raises(ValueError, match="output budget"):
        aa.check_context_limit(90, 11, 100)
    assert "each written as its headers" not in aa.NEUTRAL


def test_gap_accounts_for_correct_answers_after_retrieval_misses():
    r = {str(i): {"correct": int(i < 2)} for i in range(4)}
    a = {str(i): {"answer_correct": int(i in (0, 2))} for i in range(4)}
    c = paired_counts(r, a, list(r))
    assert c["hit_wrong"] == c["miss_correct"] == 1
    assert c["gap"] == 0 and c["conditional_em"] == 0.5


def test_answer_rescoring_and_population_mismatch_fail():
    r = record()
    a = answer(r)
    validate_answers({"q": r}, {"q": a}, "fixture")
    outside = dict(r, query_id="outside", aggregation="sum")
    validate_answers({"q": r, "outside": outside}, {"q": a}, "fixture", ids={"q"})
    a["pred"] = "999"
    with pytest.raises(ValueError, match="rescoring"):
        validate_answers({"q": r}, {"q": a}, "fixture")
    with pytest.raises(ValueError, match="IDs differ"):
        validate_answers({"q": r}, {}, "fixture")


def test_oracle_requires_explicit_matching_metadata(tmp_path):
    r = record()
    a = answer(r)
    meta = {"context_version": 2, "retrieval_records_sha256": "source",
            "query_ids_sha256": digest(["q"]), "reader_details": {"name": "fixture"},
            "prompt_sha256": "prompt", "seed": 42, "max_new_tokens": 64,
            "scorer": "hitab_exact_match_text", "excluded_unit_defect": False,
            "context_limit": 1000}
    gp, rp, op = [tmp_path / f"{name}.jsonl" for name in ("gold", "retrieved", "oracle")]
    write_pair(gp, [a], {**meta, "condition": "gold"})
    write_pair(rp, [a], {**meta, "condition": "retrieved"})
    rows, summary = compose(gp, rp, op)
    assert rows[0]["answer_correct"] == 1 and summary["composed"]
    m = json.loads(gp.with_suffix(".json").read_text())
    del m["seed"]
    gp.with_suffix(".json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="seed"):
        compose(gp, rp, tmp_path / "bad.jsonl")


def test_historical_report_uses_gold_v2_and_marks_invalid_arms():
    text = build()
    assert "t_s3c_hybrid_answer_gold_v2.jsonl" in text
    assert "Gold EM 0.9566" in text and "19.78" in text
    assert "구 RowCol 실행: 교집합" in text
    assert "2,400자" in text and "수정 코드의 성능 결과가 아닙니다" in text
    assert "온전 재현" not in text and "≈2,400자" not in text


def write_verified_fixture(tmp_path):
    r = record()
    rp = tmp_path / "run_records.jsonl"
    rm = {"context_version": 2, "comparison_scope": "index_representation_adaptation",
          "provenance": {"source_sha256": "source"}, "dataset": {"split_sha256": "data"},
          "split": "test", "corpus": "split", "alpha": 0, "budget_cells": 20,
          "budget_policy": "whole_unit_stop_after_distinct_cells", "encoder_details": None}
    write_pair(rp, [r], rm, summary_path=tmp_path / "run.json")
    from rag_agent.eval.artifacts import file_digest
    am = {"context_version": 2, "condition": "retrieved", "reader_details": {"name": "fixture"},
          "provenance": {"source_sha256": "source"}, "retrieval_records_sha256": file_digest(rp),
          "query_ids_sha256": digest(["q"]), "prompt_sha256": "prompt", "prompt": "neutral",
          "seed": 42, "max_new_tokens": 64, "scorer": "hitab_exact_match_text"}
    write_pair(tmp_path / "answers.jsonl", [answer(r)], am)
    arm = {"tag": "reference", "label": "fixture", "role": "reference", "retrieval": rp.name,
           "retrieval_summary": "run.json", "answers": "answers.jsonl"}
    spec = {"version": 2, "evidence": "verified_v2", "results_dir": ".", "reference": "reference",
            "expected_counts": {"queries": 1, "scored": 1, "primary": 1, "excluded": 0}, "arms": [arm]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path


def test_verified_report_checks_evidence_and_expected_denominators(tmp_path):
    path = write_verified_fixture(tmp_path)
    assert "검증된 새 실행" in build(path)
    spec = json.loads(path.read_text())
    spec["expected_counts"]["primary"] = 2
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="population differs"):
        load_manifest(path)


def test_verified_report_refuses_reader_context_mismatch_even_with_valid_file_hash(tmp_path):
    path = write_verified_fixture(tmp_path)
    ap = tmp_path / "answers.jsonl"
    a = read_records(ap)["q"]
    a["context_sha256"] = "different context"
    ap.write_text(json.dumps(a) + "\n")
    from rag_agent.eval.artifacts import file_digest
    mp = ap.with_suffix(".json")
    meta = json.loads(mp.read_text())
    meta["records_sha256"] = file_digest(ap)
    mp.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="reader evidence differs"):
        load_manifest(path)


def test_verified_report_refuses_uncontrolled_reader_change(tmp_path):
    path = write_verified_fixture(tmp_path)
    spec = json.loads(path.read_text())
    arm = dict(spec["arms"][0], tag="other", role="representation", answers="other.jsonl")
    meta = json.loads((tmp_path / "answers.json").read_text())
    meta["prompt_sha256"] = "different prompt"
    write_pair(tmp_path / "other.jsonl", [answer(record())], meta)
    spec["arms"].append(arm)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="uncontrolled reader setting prompt_sha256"):
        load_manifest(path)


def test_reader_main_writes_verified_evidence_without_model_download(tmp_path, monkeypatch):
    import sys
    path = write_verified_fixture(tmp_path)
    calls = []
    class Reader:
        name = "fixture"
        context_limit = 1000
        def n_prompt_tokens(self, system, user): return 20
        def complete(self, system, user, **kwargs):
            calls.append((system, user, kwargs))
            return "7"
    monkeypatch.setattr(aa, "build_llm", lambda spec: Reader())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    out = tmp_path / "reader_output.jsonl"
    monkeypatch.setattr(sys, "argv", ["answer_accuracy.py", "--records", str(tmp_path / "run_records.jsonl"),
                                      "--out", str(out)])
    assert aa.main() == 0
    result = read_records(out)["q"]
    assert len(calls) == 1 and calls[0][0] == aa.NEUTRAL
    assert result["answer_correct"] == 1 and result["cells_in_context"] == 1
    assert result["context_sha256"] == record()["context_sha256"]


def test_reader_primary_only_uses_preregistered_population(tmp_path, monkeypatch):
    import sys
    write_verified_fixture(tmp_path)
    base = record()
    extra = dict(base, query_id="outside", aggregation="sum")
    retrieval = tmp_path / "run_records.jsonl"
    with retrieval.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(extra) + "\n")
    summary = retrieval.with_name("run.json")
    meta = json.loads(summary.read_text())
    from rag_agent.eval.artifacts import file_digest
    meta["records_sha256"] = file_digest(retrieval)
    summary.write_text(json.dumps(meta))

    class Reader:
        name = "fixture"
        context_limit = 1000
        def n_prompt_tokens(self, system, user): return 20
        def complete(self, system, user, **kwargs): return "7"
        def metadata(self): return {"name": self.name}

    monkeypatch.setattr(aa, "build_llm", lambda spec: Reader())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    out = tmp_path / "primary.jsonl"
    monkeypatch.setattr(sys, "argv", ["answer_accuracy.py", "--records", str(retrieval),
                                      "--primary-only", "--out", str(out)])
    assert aa.main() == 0
    assert set(read_records(out)) == {base["query_id"]}
