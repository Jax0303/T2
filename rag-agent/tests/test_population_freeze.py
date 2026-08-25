# SPDX-License-Identifier: MIT
"""A frozen population must survive code changes, and a mixed resume must not."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag_agent.bench import population as pop_mod
from rag_agent.runenv import guard_resume


class Q:
    def __init__(self, qid):
        self.query_id = qid


class TestPin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(pop_mod, "DIR", Path(self.tmp.name))
        self.patch.start()
        pop_mod.write("p", ["a", "b", "c"], {"filter": "x"})

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_roundtrip_keeps_ids_and_meta(self):
        ids, meta = pop_mod.read("p")
        self.assertEqual(ids, ["a", "b", "c"])
        self.assertEqual(meta["filter"], "x")

    def test_pin_imposes_the_frozen_order_not_the_derived_one(self):
        """The derivation may reorder freely; the frozen file decides."""
        got = pop_mod.pin("p", [Q("c"), Q("a"), Q("b")])
        self.assertEqual([q.query_id for q in got], ["a", "b", "c"])

    def test_pin_takes_dict_queries_too(self):
        """AIT-QA and RealHiTBench build queries as dicts, HiTab as objects."""
        got = pop_mod.pin("p", [{"query_id": "c"}, {"query_id": "a"},
                                {"query_id": "b"}])
        self.assertEqual([q["query_id"] for q in got], ["a", "b", "c"])

    def test_pin_drops_queries_the_freeze_does_not_list(self):
        got = pop_mod.pin("p", [Q("a"), Q("b"), Q("c"), Q("z")])
        self.assertEqual([q.query_id for q in got], ["a", "b", "c"])

    def test_pin_refuses_a_population_that_lost_a_member(self):
        """The whole point: a code change that drops a query is an error, not a smaller n."""
        with self.assertRaises(RuntimeError):
            pop_mod.pin("p", [Q("a"), Q("b")])

    def test_unfrozen_population_passes_through(self):
        qs = [Q("x"), Q("y")]
        self.assertEqual(pop_mod.pin("not-frozen", qs), qs)


class TestGuardResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recs = Path(self.tmp.name) / "r.jsonl"
        self.env = {"seed": 42, "embed_model": "bge-small"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_call_records_the_configuration(self):
        guard_resume(self.recs, self.env, reader="local:Qwen", population="p")
        side = json.loads(Path(str(self.recs) + ".run.json").read_text())
        self.assertEqual(side["reader"], "local:Qwen")

    def test_same_configuration_resumes(self):
        guard_resume(self.recs, self.env, reader="local:Qwen", population="p")
        guard_resume(self.recs, self.env, reader="local:Qwen", population="p")

    def test_a_different_reader_is_refused(self):
        guard_resume(self.recs, self.env, reader="local:Qwen", population="p")
        with self.assertRaises(RuntimeError):
            guard_resume(self.recs, self.env, reader="groq:llama", population="p")

    def test_a_different_seed_is_refused(self):
        guard_resume(self.recs, self.env, reader="local:Qwen")
        with self.assertRaises(RuntimeError):
            guard_resume(self.recs, {**self.env, "seed": 0}, reader="local:Qwen")

    def test_force_overrides(self):
        guard_resume(self.recs, self.env, reader="local:Qwen")
        guard_resume(self.recs, self.env, reader="groq:llama", force=True)

    def test_a_different_cell_scheme_is_refused(self):
        """One embed_model, two cell sentences: the sidecar used to call these equal."""
        guard_resume(self.recs, self.env, reader="local:Qwen", cell_scheme="S3c")
        with self.assertRaises(RuntimeError):
            guard_resume(self.recs, self.env, reader="local:Qwen", cell_scheme="mt2net")

    def test_unknown_field_does_not_fabricate_a_mismatch(self):
        """A run that cannot report its population must not look like a conflict."""
        guard_resume(self.recs, self.env, reader="local:Qwen", population=None)
        guard_resume(self.recs, self.env, reader="local:Qwen", population="p")


if __name__ == "__main__":
    unittest.main()


class TestContextTableCap(unittest.TestCase):
    """A different context-table cap must not resume off another run's records."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = Path(self.tmp.name) / "r.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_different_cap_is_refused(self):
        env = {"seed": 42, "embed_model": "e"}
        guard_resume(self.rec, env, reader="x", population="p", max_context_tables=3)
        with self.assertRaises(RuntimeError):
            guard_resume(self.rec, env, reader="x", population="p",
                         max_context_tables=4)

    def test_the_same_cap_resumes(self):
        env = {"seed": 42, "embed_model": "e"}
        guard_resume(self.rec, env, reader="x", population="p", max_context_tables=3)
        guard_resume(self.rec, env, reader="x", population="p", max_context_tables=3)
