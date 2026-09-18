# SPDX-License-Identifier: MIT
"""Pinning test for scripts/colbert_rerank.py's MaxSim scorer -- pure tensor
math, no model, no data files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import math

import torch

from colbert_rerank import _insert_marker, maxsim_scores


def test_maxsim_picks_the_doc_with_the_closer_token():
    # 2-dim toy embeddings, already unit-length-ish for readability
    q_emb = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])[0]        # (Lq=2, dim=2)
    q_mask = torch.tensor([1, 1])
    # doc A has a token aligned with each query token -> high score
    d_a = [[1.0, 0.0], [0.0, 1.0]]
    # doc B is orthogonal to both query tokens -> low score
    d_b = [[-1.0, 0.0], [0.0, -1.0]]
    d_emb = torch.tensor([d_a, d_b])                            # (N=2, Ld=2, dim=2)
    d_mask = torch.tensor([[1, 1], [1, 1]])
    scores = maxsim_scores(q_emb, q_mask, d_emb, d_mask)
    assert scores[0] > scores[1]
    assert scores[0] == 2.0            # max_j sim = 1.0 for each of the 2 query tokens


def test_maxsim_ignores_padded_doc_tokens():
    q_emb = torch.tensor([[1.0, 0.0]])                          # (Lq=1, dim=2)
    q_mask = torch.tensor([1])
    # real token scores 0.1, padded token would score 1.0 if not masked
    d_emb = torch.tensor([[[0.1, 0.0], [1.0, 0.0]]])            # (N=1, Ld=2, dim=2)
    d_mask = torch.tensor([[1, 0]])                              # second token is padding
    scores = maxsim_scores(q_emb, q_mask, d_emb, d_mask)
    assert math.isclose(float(scores[0]), 0.1, abs_tol=1e-5)


def test_insert_marker_goes_right_after_cls():
    cls, sep, content, marker = 101, 102, [55, 56], 1
    ids = [cls, *content, sep]
    assert _insert_marker(ids, marker) == [cls, marker, *content, sep]


def test_maxsim_query_padding_does_not_contribute():
    q_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])               # (Lq=2, dim=2)
    q_mask = torch.tensor([1, 0])                                # second query token is padding
    d_emb = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    d_mask = torch.tensor([[1, 1]])
    scores = maxsim_scores(q_emb, q_mask, d_emb, d_mask)
    assert scores[0] == 1.0            # only the first query token counts
