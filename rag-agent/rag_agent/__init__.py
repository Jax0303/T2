"""Adaptive RAG agent over HiTab.

Architecture: original parsed 2D store + Chroma vector store, kept separate.
Stages are routed by query type (rule-based on HiTab paper's aggregation
taxonomy) and the agent skips stages that aren't needed.
"""
__version__ = "0.1.0"
