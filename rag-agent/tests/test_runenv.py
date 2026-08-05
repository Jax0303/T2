# SPDX-License-Identifier: MIT
"""run_env seeds the RNGs it claims to seed, and emits the documented block."""
import random

import numpy as np

from rag_agent.runenv import run_env


def test_seeds_are_actually_set():
    env = run_env(42, "BAAI/bge-small-en-v1.5")
    a = (random.random(), float(np.random.rand()))
    assert run_env(42, "x") is not None
    b = (random.random(), float(np.random.rand()))
    assert a == b
    assert env["seed"] == 42
    assert env["embed_model"] == "BAAI/bge-small-en-v1.5"
    assert set(env) == {"seed", "device", "torch", "embed_model", "run_started_utc"}
