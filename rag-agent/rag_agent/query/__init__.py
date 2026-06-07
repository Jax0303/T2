# SPDX-License-Identifier: MIT
"""Query-understanding layer.

Resolves a natural-language table question into a *hierarchical header-path
intent* — the shared intermediate representation between the (separate)
vector store and the original header-tree store.
"""
from .header_path_resolver import (  # noqa: F401
    HeaderPathIntent,
    extract_target_terms,
    expand_for_retrieval,
    resolve_against_table,
    resolve_intent,
)
