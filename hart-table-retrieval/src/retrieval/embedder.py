import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_dim(self) -> int:
        pass


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._model_name = model_name
        self._client = OpenAI()
        self._dim = 1536

    def embed(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        batch_size = 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(input=batch, model=self._model_name)
            all_embeddings.extend([d.embedding for d in resp.data])
            if i + batch_size < len(texts):
                time.sleep(0.1)
        return np.array(all_embeddings, dtype=np.float32)

    def get_name(self) -> str:
        return self._model_name

    def get_dim(self) -> int:
        return self._dim


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: str = None):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._batch_size = 128 if device == "cuda" else 32
        logger.info("Loaded %s on %s (batch_size=%d)", model_name, device, self._batch_size)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self._device,
        )
        return embeddings.astype(np.float32)

    def get_name(self) -> str:
        return self._model_name

    def get_dim(self) -> int:
        return self._dim


class OnnxDefaultEmbedder(BaseEmbedder):
    """Lightweight fallback using ChromaDB's onnx-based default embedder
    (all-MiniLM-L6-v2, 384 dim). Used when sentence-transformers is unavailable.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()
        self._dim = 384
        self._name = "chromadb-default-onnx"

    def embed(self, texts: List[str]) -> np.ndarray:
        embs = self._fn(texts)
        return np.array(embs, dtype=np.float32)

    def get_name(self) -> str:
        return self._name

    def get_dim(self) -> int:
        return self._dim


class EmbedderFactory:
    @staticmethod
    def create(config: dict) -> BaseEmbedder:
        if config["type"] == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set - skipping OpenAI embedder '%s'",
                    config["name"],
                )
                return None
            return OpenAIEmbedder(config["name"])
        elif config["type"] == "sentence-transformer":
            try:
                return SentenceTransformerEmbedder(config["name"], device=config.get("device"))
            except ImportError as e:
                logger.warning(
                    "sentence-transformers package missing (%s), falling back to onnx default",
                    e,
                )
                return OnnxDefaultEmbedder()
            except Exception as e:
                logger.warning(
                    "sentence-transformer model '%s' failed to load (%s: %s), skipping",
                    config["name"], type(e).__name__, str(e)[:200],
                )
                return None
        elif config["type"] == "onnx-default":
            return OnnxDefaultEmbedder()
        else:
            raise ValueError(f"Unknown embedder type: {config['type']}")
