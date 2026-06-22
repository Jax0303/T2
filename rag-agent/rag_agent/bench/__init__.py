# SPDX-License-Identifier: MIT
"""Benchmark-agnostic layer for the operand-targeted RAG pipeline.

Three benchmarks (HiTab, FinQA, WikiSQL) are mapped into one schema
(:class:`~rag_agent.bench.schema.BenchTable` / ``BenchQuery`` / ``GoldOperand``)
so every downstream stage — serialization, HPIR decomposition, operand-targeted
retrieval, coverage/fallback, generation — is written once and runs unchanged on
all three. See ``docs/OPERAND_RAG.md``.
"""
from .schema import BenchTable, BenchQuery, GoldOperand, Chunk

__all__ = ["BenchTable", "BenchQuery", "GoldOperand", "Chunk"]
