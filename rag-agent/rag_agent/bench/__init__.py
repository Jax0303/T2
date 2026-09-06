# SPDX-License-Identifier: MIT
"""HiTab (and the MultiHiertt document corpus) mapped into one schema
(:class:`~rag_agent.bench.schema.BenchTable` / ``BenchQuery`` / ``GoldOperand``).
Frozen query populations live in :mod:`.population`."""
from .schema import BenchTable, BenchQuery, GoldOperand, Chunk

__all__ = ["BenchTable", "BenchQuery", "GoldOperand", "Chunk"]
