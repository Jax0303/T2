# SPDX-License-Identifier: MIT
"""HyDE — Hypothetical Document Embeddings (Gao et al. 2022, arXiv:2212.10496).

Ask an LLM to invent the passage that WOULD answer the query, then retrieve with
that passage's embedding instead of the query's. The query and the corpus are
written in different registers -- a question versus a table cell rendered as a
statement -- and a bi-encoder has to bridge that gap inside one dot product.
A hypothetical passage is already in the corpus's register, so the comparison
becomes document-to-document.

Why this repo runs it: this work's claim is that the fix belongs on the DOCUMENT
side (how a cell becomes text). HyDE is the strongest cheap intervention on the
QUERY side. If HyDE closes the gap, the claim does not survive, so this arm has
to be run before the document-side arms mean anything -- and it has to be run at
the paper's own settings, not a crippled version, or the negative result is
worthless.

Faithful to the paper on the two settings that matter:

* ``n`` generations per query, sampled at nonzero temperature, and the final
  vector is the mean of the generated passages AND the query itself
  (``1/(N+1) [sum f(d_k) + f(q)]``, §3). The mean is taken by the CALLER, which
  owns the encoder; this module only produces text.
* a task-specific instruction. The paper writes one per dataset (web search,
  SciFact, TREC-COVID each get their own), so naming the table-cell target here
  is following their recipe rather than tuning our own.

Default ``n=1`` against the paper's 8: one call per query is enough to see
whether the effect exists at all, and 8x the cost only matters once it does.
Raise it before publishing a negative result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SYSTEM = (
    "You write short passages that imitate the prose of a statistical report, "
    "for use as retrieval probes."
)

INSTRUCTION = (
    "Write a short passage that states the facts needed to answer the question, "
    "as if copied from a statistical table. Write each fact as its own sentence "
    "naming the row and column headings it sits under, then its value. "
    "Invent plausible values -- the passage is a retrieval probe, not an answer. "
    "No preamble, no explanation.\n\n"
    "Question: {q}\nPassage:"
)


class Hyde:
    """Generates hypothetical passages, cached on disk by (prompt, draw).

    ``n > 1`` only produces distinct draws if ``llm`` was BUILT with a nonzero
    temperature -- this repo's LLM interface fixes temperature on the backend,
    not per call. The caller is responsible for that; ``n > 1`` against a
    temperature-0 backend would silently average N copies of one passage.
    """

    def __init__(self, llm, cache_path: Optional[str] = None, n: int = 1,
                 max_tokens: int = 160):
        self.llm = llm
        self.n = n
        self.max_tokens = max_tokens
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self.cache[rec["key"]] = rec["passage"]

    def _key(self, question: str, draw: int) -> str:
        # the prompt is part of the key: editing it must invalidate every cached
        # passage, or a run silently mixes two prompt versions
        h = hashlib.sha1(
            f"{SYSTEM}\x1f{INSTRUCTION}\x1f{question}\x1f{draw}".encode())
        return h.hexdigest()

    def _generate(self, question: str, draw: int) -> str:
        key = self._key(question, draw)
        if key in self.cache:
            return self.cache[key]
        passage = self.llm.complete(
            SYSTEM, INSTRUCTION.format(q=question), max_tokens=self.max_tokens
        ).strip()
        if not passage:
            # a refusal or an empty completion must not silently become an empty
            # query -- fall back to the question so the arm degrades to no-HyDE
            # for that one query instead of scoring a blank probe
            passage = question
        self.cache[key] = passage
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as fh:
                fh.write(json.dumps({"key": key, "question": question,
                                     "draw": draw, "passage": passage}) + "\n")
        return passage

    def passages(self, questions: Sequence[str]) -> List[List[str]]:
        """``questions[i]`` -> its ``n`` hypothetical passages."""
        return [[self._generate(q, d) for d in range(self.n)] for q in questions]
