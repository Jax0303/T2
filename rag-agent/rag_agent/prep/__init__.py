"""Pre-retrieval table preprocessing conditions (C0–C3) and evaluation utilities.

Implements the controlled "preprocessing gain × table complexity" experiment:
the same cumulative augmentation ladder (raw → +metadata → +schema description
→ +synthetic questions) applied to a flat corpus (OpenWikiTable) and a
hierarchical corpus (HiTab), measured by retrieval recall@k.
"""

from rag_agent.prep.conditions import PrepTable, serialize, CONDITIONS
from rag_agent.prep.stats import recall_at_k, paired_delta_bootstrap

__all__ = [
    "PrepTable",
    "serialize",
    "CONDITIONS",
    "recall_at_k",
    "paired_delta_bootstrap",
]
