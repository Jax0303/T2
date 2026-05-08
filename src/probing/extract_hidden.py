# SPDX-License-Identifier: MIT
"""Extract layer-wise hidden states from sentence-transformer models.

Supports BGE-small-en-v1.5 and E5-small-v2 via forward hooks on each
transformer layer.  Returns hidden states for every layer as numpy arrays.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class PoolingStrategy(str, Enum):
    """How to reduce token-level hidden states to a single vector."""

    CLS = "cls"
    MEAN = "mean"


# Pre-defined model names.
MODEL_REGISTRY: dict[str, str] = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
}


class HiddenStateExtractor:
    """Extract per-layer hidden states from a transformer encoder.

    Args:
        model_key: Short name (``bge-small`` or ``e5-small``) or a
            HuggingFace model id.
        pooling: Pooling strategy (``cls`` or ``mean``).
        device: Torch device string.  Defaults to ``cpu``.
    """

    def __init__(
        self,
        model_key: str = "bge-small",
        pooling: PoolingStrategy | str = PoolingStrategy.CLS,
        device: str = "cpu",
    ) -> None:
        model_name = MODEL_REGISTRY.get(model_key, model_key)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.pooling = PoolingStrategy(pooling)

        # Discover number of layers.
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            self.n_layers: int = len(encoder.layer)
        else:
            # Fallback: count config hidden layers.
            self.n_layers = int(self.model.config.num_hidden_layers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Extract hidden states for a list of texts.

        Args:
            texts: Input strings.
            batch_size: Tokenisation / forward batch size.

        Returns:
            Array of shape ``(n_texts, n_layers + 1, hidden_dim)``.
            Index 0 is the embedding layer output; indices 1..n_layers are
            the transformer layer outputs.
        """
        all_hidden: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                )

            # outputs.hidden_states: tuple of (n_layers+1) tensors,
            # each (batch, seq_len, hidden_dim).
            hidden_states = outputs.hidden_states  # type: ignore[attr-defined]
            attention_mask = encoded["attention_mask"]

            # Pool each layer.
            batch_layers: list[np.ndarray] = []
            for layer_tensor in hidden_states:
                pooled = self._pool(layer_tensor, attention_mask)
                batch_layers.append(pooled.cpu().numpy())

            # Stack: (n_layers+1, batch, hidden_dim) → (batch, n_layers+1, hidden_dim)
            stacked = np.stack(batch_layers, axis=0)  # (L+1, B, D)
            stacked = stacked.transpose(1, 0, 2)  # (B, L+1, D)
            all_hidden.append(stacked)

        return np.concatenate(all_hidden, axis=0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pool(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling to a (batch, seq_len, dim) tensor."""
        if self.pooling == PoolingStrategy.CLS:
            return hidden[:, 0, :]
        # Mean pooling.
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return summed / lengths
