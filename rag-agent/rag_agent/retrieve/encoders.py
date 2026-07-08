"""Encoders for dense retrieval, with a dependency-free fallback.

The thesis uses a real sentence encoder (BGE) on the GPU box, but the package
must also import and run where ``torch`` / ``sentence-transformers`` are absent
(CI, a fresh CPU container). So the dense backend is pluggable:

* :class:`SentenceTransformerEncoder` — the real BGE encoder, lazily imported.
* :class:`HashingEncoder` — a deterministic hashed bag-of-words TF encoder using
  only NumPy. Lexical, not semantic, but enough to exercise the retrieval
  plumbing and unit tests anywhere.

Both return L2-normalized row vectors so a dot product is cosine similarity.

Embedding-consistency rule (from the prompt spec): the **same encoder must embed
both the chunks and the queries**. The retriever enforces this by holding a
single encoder instance for both.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_&$%/.-]*")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


@runtime_checkable
class Encoder(Protocol):
    """Encodes a list of strings into an (n, dim) L2-normalized float array."""

    name: str

    def encode(self, texts: List[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEncoder:
    """Deterministic hashed bag-of-words TF encoder (NumPy only, no model).

    Sublinear term frequency + the hashing trick into ``dim`` buckets, then L2
    normalization. Purely lexical; used as a fallback and in tests so the
    pipeline runs without heavyweight ML dependencies.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing_tf_{dim}"

    def _bucket(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    def encode(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokenize(t):
                mat[i, self._bucket(tok)] += 1.0
        # sublinear tf
        np.log1p(mat, out=mat)
        return _l2_normalize(mat)


class SentenceTransformerEncoder:
    """Real dense encoder (e.g. BGE). Lazily loads ``sentence-transformers``."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy
        import torch

        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


def default_encoder(prefer_model: bool = True, **kwargs) -> Encoder:
    """Return a real ST encoder if its deps are importable, else HashingEncoder."""
    if prefer_model:
        try:
            return SentenceTransformerEncoder(**kwargs)
        except Exception:
            pass
    return HashingEncoder()
