# SPDX-License-Identifier: MIT
"""YAML-config merging in scripts/run_eval.py.

Loaded by path because ``scripts/`` is not an importable package.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_eval)

CFG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _merged(argv):
    args = run_eval.build_arg_parser().parse_args(argv)
    return run_eval._merge_config(args, argv)


class TestMergeConfig(unittest.TestCase):
    BASE = ["--config", str(CFG_DIR / "v3.1_baseline.yaml"),
            "--data-dir", "d", "--chroma-dir", "c"]

    def test_config_applies_when_cli_silent(self):
        args = _merged(self.BASE)
        self.assertEqual(args.llm, "local:Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(args.per_class, 8)

    def test_cli_wins_even_when_value_equals_the_parser_default(self):
        """The old rule was 'override only if the value still equals the
        default', which silently discarded a flag you passed explicitly when
        you happened to pass the default value."""
        argv = self.BASE + ["--llm", "groq:llama-3.3-70b-versatile"]
        self.assertEqual(_merged(argv).llm, "groq:llama-3.3-70b-versatile")

    def test_equals_form_is_recognised(self):
        self.assertEqual(_merged(self.BASE + ["--per-class=3"]).per_class, 3)

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.yaml"
            bad.write_text("w_verify: 0.3\n")           # a flag that no longer exists
            argv = ["--config", str(bad), "--data-dir", "d", "--chroma-dir", "c"]
            with self.assertRaises(SystemExit) as ctx:
                _merged(argv)
            self.assertIn("w_verify", str(ctx.exception))

    def test_shipped_configs_all_load(self):
        for cfg in sorted(CFG_DIR.glob("*.yaml")):
            if cfg.name in {"hpir.yaml", "prep.yaml"}:   # not run_eval configs
                continue
            with self.subTest(config=cfg.name):
                _merged(["--config", str(cfg), "--data-dir", "d", "--chroma-dir", "c"])


if __name__ == "__main__":
    unittest.main()
