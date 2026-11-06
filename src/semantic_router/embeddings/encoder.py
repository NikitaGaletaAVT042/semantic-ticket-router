from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from src.semantic_router.embeddings.vector_match import l2_normalize


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


class HashingEncoder(BaseEncoder):
    """
    Легкий deterministic encoder для тестов и локального smoke-test.

    Это НЕ замена sentence-transformers для продакшена.
    Он нужен, чтобы тесты не тянули тяжелую модель.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest[:8], 16) % self.dim
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vectors[row_idx, idx] += sign

        return l2_normalize(vectors)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.lower().replace("ё", "е")
        return [part.strip(".,!?;:()[]{}\"'«»") for part in normalized.split() if part.strip()]