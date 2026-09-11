"""Encoders for dense retrieval, with a dependency-free fallback.

The thesis uses a real sentence encoder (BGE) on the GPU box, but the package
must also import and run where ``torch`` / ``sentence-transformers`` are absent
(CI, a fresh CPU container). So the dense backend is pluggable:

* :class:`SentenceTransformerEncoder` — the real BGE encoder, lazily imported.
* :class:`HashingEncoder` — a deterministic hashed bag-of-words TF encoder using
  only NumPy. Lexical, not semantic, but enough to exercise the retrieval
  plumbing and unit tests anywhere. Selecting it is always explicit: either
  construct it directly, or pass ``allow_fallback=True`` to
  :func:`default_encoder`. It is never substituted for a failed model load
  behind the caller's back — see that function's docstring.

Both return L2-normalized row vectors so a dot product is cosine similarity.

Embedding-consistency rule: the **same encoder must embed both the chunks and
the queries**. The retriever enforces this by holding a single encoder instance
for both. Provenance of that encoder is a reproducibility item — see
``RESEARCH_STRUCTURE.md`` §6 (재현성 부채: 임베딩 모델 리비전 / 실행 환경), which is
also why the hashing fallback is never substituted silently.
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

    def encode_query(self, texts: List[str]) -> np.ndarray: ...


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
        prefixes: Optional[tuple] = None,
        revision: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy
        import torch

        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device,
                                         revision=revision, trust_remote_code=True)
        self.revision = revision
        self.batch_size = batch_size
        # The asymmetry the model was trained with. Resolved here, from the
        # model's own name, so no caller can forget it -- see _QUERY_PREFIXES.
        self.query_prefix, self.passage_prefix = (
            prefixes if prefixes is not None else default_prefixes(model_name))

    def _encode(self, texts: List[str], prefix: str) -> np.ndarray:
        vecs = self.model.encode(
            [prefix + t for t in texts] if prefix else texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)

    def encode_query(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def metadata(self) -> dict:
        first = self.model[0]
        config = getattr(getattr(first, "auto_model", None), "config", None)
        return {"name": self.name, "revision_requested": self.revision,
                "revision_resolved": getattr(config, "_commit_hash", None),
                "max_seq_length": int(self.model.max_seq_length),
                "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
                "device": self.device, "batch_size": self.batch_size,
                "normalized": True}

    def audit_inputs(self, texts, *, query=False, overflow="error") -> dict:
        """Count full prefixed inputs, including special tokens, before encoding.

        The tokenizer call deliberately disables truncation. A caller must opt
        into the model's normal truncation and retain the returned audit.
        """
        if overflow not in {"error", "truncate"}:
            raise ValueError("overflow must be error or truncate")
        prefix = self.query_prefix if query else self.passage_prefix
        lengths = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[start:start + self.batch_size]]
            tokens = self.model.tokenizer(batch, truncation=False, padding=False,
                                          add_special_tokens=True)["input_ids"]
            lengths.extend(map(len, tokens))
        limit = int(self.model.max_seq_length)
        over = [i for i, n in enumerate(lengths) if n > limit]
        report = {"n": len(lengths), "max_tokens": max(lengths, default=0),
                  "max_seq_length": limit, "n_overflow": len(over),
                  "overflow_ratio": len(over) / len(lengths) if lengths else 0,
                  "overflow_indices": over, "policy": overflow}
        if over and overflow == "error":
            raise ValueError(f"encoder input overflow: {len(over)}/{len(lengths)} "
                             f"exceed {limit} tokens (max={max(lengths)}); "
                             "choose a suitable encoder/chunk size or explicitly "
                             "allow and report --embed-overflow truncate")
        return report


# Instruction prefixes the embedder families were TRAINED with. Retrieval is
# asymmetric -- a question and the passage answering it are not paraphrases --
# and these models learn that asymmetry from a prefix on one side only. Encoding
# both sides bare is a silent misuse: it still runs, still returns plausible
# numbers, and leaves points on the floor for every arm at once.
#
# BGE English v1.5 puts an instruction on the QUERY and nothing on the passage.
# E5 marks both sides. bge-m3 and the gte family were trained without prefixes,
# so anything not listed here gets none rather than a guess.
_QUERY_PREFIXES = {
    "bge-": ("Represent this sentence for searching relevant passages: ", ""),
    "e5-": ("query: ", "passage: "),
}


def default_prefixes(model_name: str) -> tuple:
    """``(query_prefix, passage_prefix)`` for ``model_name``, ``("", "")`` if unknown.

    Callers pass the encoder's OWN name (``encoder.name``), not the requested
    model string -- see :func:`default_encoder` on why those can differ.
    """
    low = (model_name or "").lower()
    if "bge-m3" in low:            # trained without an instruction, unlike v1.5
        return ("", "")
    for family, pair in _QUERY_PREFIXES.items():
        if family in low:
            return pair
    return ("", "")


def default_encoder(
    prefer_model: bool = True,
    allow_fallback: bool = False,
    **kwargs,
) -> Encoder:
    """Return a real ST encoder, or :class:`HashingEncoder` if asked to.

    The fallback is **opt-in**. A silent downgrade used to be possible here, and
    because a run only recorded the *requested* model name, a hashed-bag-of-words
    run and a real BGE run were indistinguishable in the output JSON. Reported
    dense numbers must be auditable, so a model that fails to load now raises;
    pass ``allow_fallback=True`` (or ``prefer_model=False``) to accept the
    lexical encoder deliberately. Callers writing result files should record
    ``encoder.name``, not the requested model string.
    """
    if prefer_model:
        try:
            return SentenceTransformerEncoder(**kwargs)
        except Exception:
            if not allow_fallback:
                raise
    return HashingEncoder()
