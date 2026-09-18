#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan item 4: ColBERT-style late-interaction rerank of the deployed
hybrid's top-K pool. [[leaf-boost-rerank-closed-2026-09-17]] closed the flat
literal-leaf-token bonus (too non-selective at corpus scale);
[[cross-encoder-rerank-closed-2026-09-17]] closed the joint cross-encoder
(bge-reranker-v2-m3, no significant effect). This tries a third, structurally
different mechanism -- token-level MaxSim instead of one pooled vector or one
joint [CLS] score -- since neither prior arm directly addressed "a short
leaf token drowned inside a much longer shared title/header string."

colbert-ir/colbertv2.0 -- inference only, no training. Its HF checkpoint
declares a custom "HF_ColBERT" architecture (plain BertModel + a 128-dim
linear projection, no bias) that AutoModel cannot build from the
architecture name alone, so this loads the base BertModel via its
model_type="bert" field and reattaches `linear.weight` from the same
checkpoint file by hand. No colbert-ai/RAGatouille dependency and no PLAID
index -- MaxSim is a few lines, and this only reranks an already-generated
top-50 pool, never a million-document index.

Uses the paper's own preprocessing (Khattab & Zaharia 2020, SS3): a [Q]
marker ([unused0]) right after [CLS] on the query side, padded to
QUERY_MAXLEN=32 with [MASK] (not [PAD]) and attention_mask forced to all-1s
-- "query augmentation," the mask positions ARE attended over, not ignored --
and a [D] marker ([unused1]) after [CLS] on the doc side, padded normally to
DOC_MAXLEN=180 with [PAD]/attention_mask=0, plus a punctuation skiplist
(masked out of MaxSim same as padding). A first pass (see [[colbert-rerank-
closed-2026-09-17]]) skipped all of this and got a large regression (R@1
.62->.50, p=~0); this redo isolates whether that was the mechanism or the
missing preprocessing.

Pool = hybrid top-50 (same deployed structural_leaf hybrid, ALPHA=0.7, as
every other script in this investigation -- leaf_boost_rerank.build_corpus,
reused as-is). K=20 and K=50 both read off ONE MaxSim pass per query, same
convention as cross_encoder_rerank.py, so there is no dev-side K selection
to prereg or overfit.

  PYTHONPATH=. .venv/bin/python scripts/colbert_rerank.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402
from scipy.stats import binomtest                                     # noqa: E402

from rag_agent.retrieve.encoders import default_encoder                # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax, _tokenize         # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                  # noqa: E402
from bottleneck_diagnosis import load_primary_population, split_corpus_table_ids  # noqa: E402
from leaf_boost_rerank import build_corpus                            # noqa: E402
from sentence_disambiguation_eval import encode_corpus                # noqa: E402
from cross_encoder_rerank import has_cuda, mcnemar                    # noqa: E402

OUT_DIR = ROOT / "results/colbert_rerank_20260917"
ALPHA = 0.7                     # same deployed hybrid weight as embedding_fusion_diagnosis.py
POOL_SIZE = 50
COLBERT_MODEL = "colbert-ir/colbertv2.0"
QUERY_MAXLEN = 32               # official ColBERT default
DOC_MAXLEN = 180                # official ColBERT default


def maxsim_scores(q_emb: torch.Tensor, q_mask: torch.Tensor,
                  d_emb: torch.Tensor, d_mask: torch.Tensor) -> np.ndarray:
    """ColBERT's late-interaction score: one query (Lq, dim) against N docs
    (N, Ld, dim) -> (N,) scores, ``sum_i max_j (q_i . d_j)`` over query
    positions ``i`` and doc positions ``j`` (padding masked out of the max on
    the doc side, and out of the sum via ``q_mask`` on the query side). Pure
    tensor math -- no model needed, so it is unit-tested directly."""
    sim = torch.einsum("qd,nkd->nqk", q_emb, d_emb)
    sim = sim.masked_fill(d_mask.unsqueeze(1) == 0, -1e4)
    per_query_tok_max = sim.max(dim=2).values          # (N, Lq)
    return (per_query_tok_max * q_mask.unsqueeze(0)).sum(dim=1).cpu().numpy()


def _build_skiplist(tok) -> set:
    """Single-token punctuation ids -- masked out of doc-side MaxSim the same
    way padding is, per the paper's own preprocessing."""
    import string
    ids = set()
    for ch in string.punctuation:
        enc = tok.encode(ch, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def _insert_marker(ids: list, marker_id: int) -> list:
    return [ids[0], marker_id] + ids[1:]


class ColbertScorer:
    """BertModel + the checkpoint's own 128-dim projection, loaded by hand
    (see module docstring for why AutoModel alone cannot build this), with
    the paper's query/doc tokenization (marker tokens, query-side [MASK]
    augmentation, doc-side punctuation skiplist)."""

    def __init__(self, device: str):
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(COLBERT_MODEL)
        self.bert = AutoModel.from_pretrained(COLBERT_MODEL, add_pooling_layer=False).to(device).eval()
        weights_path = hf_hub_download(COLBERT_MODEL, "model.safetensors")
        lin_w = load_file(weights_path)["linear.weight"]
        self.linear = torch.nn.Linear(lin_w.shape[1], lin_w.shape[0], bias=False).to(device).eval()
        with torch.no_grad():
            self.linear.weight.copy_(lin_w)
        self.q_marker = self.tok.convert_tokens_to_ids("[unused0]")
        self.d_marker = self.tok.convert_tokens_to_ids("[unused1]")
        self.skiplist = _build_skiplist(self.tok)

    def _run(self, ids_batch: list, masks_batch: list):
        maxlen = max(len(ids) for ids in ids_batch)
        pad_id = self.tok.pad_token_id
        ids = torch.tensor([row + [pad_id] * (maxlen - len(row)) for row in ids_batch])
        mask = torch.tensor([row + [0] * (maxlen - len(row)) for row in masks_batch])
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            proj = F.normalize(self.linear(self.bert(input_ids=ids,
                                                      attention_mask=mask).last_hidden_state), dim=-1)
        return proj, mask

    def encode_query(self, texts: list):
        """[CLS] [Q] tokens... padded to QUERY_MAXLEN with [MASK]; mask is
        all-1s (query augmentation attends over the [MASK] padding too)."""
        mask_id = self.tok.mask_token_id
        rows = []
        for t in texts:
            ids = self.tok.encode(t, add_special_tokens=True,
                                  truncation=True, max_length=QUERY_MAXLEN - 1)
            ids = _insert_marker(ids, self.q_marker)
            ids = ids + [mask_id] * (QUERY_MAXLEN - len(ids))
            rows.append(ids[:QUERY_MAXLEN])
        masks = [[1] * QUERY_MAXLEN for _ in rows]
        return self._run(rows, masks)

    def encode_doc(self, texts: list):
        """[CLS] [D] tokens... [SEP], normally padded; punctuation-only
        tokens get attention_mask=0 (skiplist), same treatment as padding."""
        rows = []
        for t in texts:
            ids = self.tok.encode(t, add_special_tokens=True,
                                  truncation=True, max_length=DOC_MAXLEN - 1)
            rows.append(_insert_marker(ids, self.d_marker))
        masks = [[0 if tid in self.skiplist else 1 for tid in row] for row in rows]
        return self._run(rows, masks)


def rerank_hits(pop: dict, texts: list, coord_index: dict, emb, bm, enc, colbert,
                limit: int = 0) -> dict:
    base_hits, k20_hits, k50_hits = {}, {}, {}
    ordered = sorted(pop.items())
    if limit:
        ordered = ordered[:limit]
    for qid, r in ordered:
        gold = tuple(sorted(r["gold_cells"])[0])
        gidx = coord_index.get(gold)
        if gidx is None:
            continue
        q = r["question"]
        dense = emb @ enc.encode_query([q])[0].astype(np.float32)
        sparse = bm.get_scores(_tokenize(q))
        hybrid = ALPHA * _minmax(dense) + (1 - ALPHA) * _minmax(sparse)
        pool = np.argsort(-hybrid, kind="stable")[:POOL_SIZE]

        q_emb, q_mask = colbert.encode_query([q])
        d_emb, d_mask = colbert.encode_doc([texts[i] for i in pool])
        cb_scores = maxsim_scores(q_emb[0], q_mask[0], d_emb, d_mask)

        base_hits[qid] = bool(pool[0] == gidx)
        k20_hits[qid] = bool(pool[:20][int(np.argmax(cb_scores[:20]))] == gidx)
        k50_hits[qid] = bool(pool[int(np.argmax(cb_scores))] == gidx)
    return base_hits, k20_hits, k50_hits


def run(data_dir: str = "data/hitab", limit: int = 0) -> dict:
    pop = load_primary_population()
    tids = split_corpus_table_ids()
    texts, _row_leaf, _col_leaf, coords = build_corpus(tids, data_dir)
    coord_index = {c: i for i, c in enumerate(coords)}

    enc = default_encoder()
    emb, cache_hit = encode_corpus(texts, enc)
    bm = SparseBM25(_tokenize(t) for t in texts)
    colbert = ColbertScorer(device="cuda" if has_cuda() else "cpu")

    t0 = time.time()
    base_hits, k20_hits, k50_hits = rerank_hits(pop, texts, coord_index, emb, bm, enc, colbert, limit)
    elapsed = time.time() - t0

    summary = {
        "n": len(base_hits), "pool_size": POOL_SIZE, "rerank_model": COLBERT_MODEL,
        "preprocessing": "official (Q/D markers, query MASK augmentation, doc skiplist)",
        "elapsed_s": round(elapsed, 1), "embed_cache_hit": cache_hit,
        "baseline_vs_k20": mcnemar(base_hits, k20_hits, "baseline", "k20"),
        "baseline_vs_k50": mcnemar(base_hits, k50_hits, "baseline", "k50"),
        "k20_vs_k50": mcnemar(k20_hits, k50_hits, "k20", "k50"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    raise SystemExit(0 if run(limit=n) else 1)
