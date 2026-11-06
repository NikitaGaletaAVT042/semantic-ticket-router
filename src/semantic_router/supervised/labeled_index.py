from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.semantic_router.preprocessing.normalizer import normalize_text


class EncoderProtocol(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class LabeledExample:
    key: str
    label: str
    text: str


@dataclass(frozen=True)
class Neighbor:
    key: str
    label: str
    score: float
    text: str


class LabeledKnnIndex:
    def __init__(
        self,
        *,
        examples: list[LabeledExample],
        vectors: np.ndarray,
    ) -> None:
        self.examples = examples
        self.vectors = _normalize_matrix(vectors)

    @classmethod
    def build(
        cls,
        *,
        examples: list[LabeledExample],
        encoder: EncoderProtocol,
    ) -> "LabeledKnnIndex":
        texts = [normalize_text(example.text) for example in examples]
        vectors = encoder.encode(texts)

        return cls(
            examples=examples,
            vectors=np.asarray(vectors, dtype=np.float32),
        )

    def search_with_vector(
        self,
        *,
        query_vector: np.ndarray,
        top_k: int = 7,
    ) -> list[Neighbor]:
        if not self.examples:
            return []

        query = np.asarray(query_vector, dtype=np.float32)

        if query.ndim == 2:
            query = query[0]

        query = _normalize_vector(query)

        scores = self.vectors @ query

        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [
            Neighbor(
                key=self.examples[index].key,
                label=self.examples[index].label,
                score=float(scores[index]),
                text=self.examples[index].text,
            )
            for index in top_indices
        ]

    def vote(
        self,
        *,
        query_vector: np.ndarray,
        top_k: int = 7,
    ) -> tuple[str | None, float, list[Neighbor]]:
        neighbors = self.search_with_vector(
            query_vector=query_vector,
            top_k=top_k,
        )

        if not neighbors:
            return None, 0.0, []

        label_scores: defaultdict[str, float] = defaultdict(float)

        for neighbor in neighbors:
            label_scores[neighbor.label] += max(0.0, neighbor.score)

        best_label, best_score = max(
            label_scores.items(),
            key=lambda item: item[1],
        )

        total_score = sum(label_scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.0

        return best_label, confidence, neighbors


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return vectors / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)

    norm = np.linalg.norm(vector)
    norm = max(float(norm), 1e-12)

    return vector / norm