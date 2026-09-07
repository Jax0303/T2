"""rag_agent -- RAG over hierarchical-header tables.

Index unit: one data cell -> one sentence (title + row path + column path +
value), embedded and ranked corpus-wide by a hybrid BM25 + dense index
(:mod:`rag_agent.retrieve`). Benchmarks load through :mod:`rag_agent.bench`.
"""
__version__ = "0.4.0"
