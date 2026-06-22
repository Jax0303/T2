# SPDX-License-Identifier: MIT
"""Table serialization for the operand-targeted pipeline.

Two schemes, both row-level (one chunk per data row) so a retrieved chunk maps
back to concrete cells for ``operand_recall``:

  * ``S1`` (flat)               — leaf headers only; the structure-naive control.
  * ``S2`` (structure-preserving) — each cell prefixed with its full header path
    ``left_path > top_path: value`` (e.g. ``Revenue > 2023 > Q1: 1,234``).

The S1↔S2 gap isolates the contribution of hierarchical header structure (the
"S1 ablation" in ``docs/OPERAND_RAG.md``).
"""
from .serializers import serialize_table, fulltable_chunk, S1, S2, SCHEMES

__all__ = ["serialize_table", "fulltable_chunk", "S1", "S2", "SCHEMES"]
