from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.semantic_router.embeddings.encoder import BaseEncoder
from src.semantic_router.embeddings.vector_match import l2_normalize
from src.semantic_router.taxonomy.card_builder import GroupCard, TopicCard


@dataclass(frozen=True)
class TopicEmbeddingRecord:
    group_id: str
    group_name: str
    topic_id: str
    topic_name: str
    status: str
    positive_vectors: np.ndarray
    negative_vectors: np.ndarray | None


@dataclass(frozen=True)
class GroupEmbeddingRecord:
    group_id: str
    group_name: str
    vector: np.ndarray


@dataclass(frozen=True)
class EmbeddingIndex:
    groups: list[GroupEmbeddingRecord]
    topics: list[TopicEmbeddingRecord]

    def topics_by_group(self, group_id: str) -> list[TopicEmbeddingRecord]:
        return [topic for topic in self.topics if topic.group_id == group_id]


def build_embedding_index(
    *,
    encoder: BaseEncoder,
    topic_cards: list[TopicCard],
    group_cards: list[GroupCard],
) -> EmbeddingIndex:
    group_records = _build_group_records(encoder, group_cards)
    topic_records = _build_topic_records(encoder, topic_cards)

    return EmbeddingIndex(
        groups=group_records,
        topics=topic_records,
    )


def _build_group_records(
    encoder: BaseEncoder,
    group_cards: list[GroupCard],
) -> list[GroupEmbeddingRecord]:
    if not group_cards:
        return []

    texts = [card.card_text for card in group_cards]
    vectors = encoder.encode(texts)

    records: list[GroupEmbeddingRecord] = []
    for card, vector in zip(group_cards, vectors):
        records.append(
            GroupEmbeddingRecord(
                group_id=card.group_id,
                group_name=card.group_name,
                vector=vector,
            )
        )

    return records

def _build_positive_texts_for_topic(card: TopicCard) -> list[str]:
    """
    Строит набор независимых positive-прототипов для темы.

    Важно:
    - не склеиваем всё в один огромный текст;
    - topic_name даём отдельно;
    - topic_description даём отдельно;
    - добавляем связку group + topic, чтобы модель лучше понимала контекст;
    - positive_examples оставляем отдельными векторами.
    """
    texts = [
        card.topic_name,
        f"{card.group_name}. {card.topic_name}",
        card.topic_description,
        f"{card.topic_name}. {card.topic_description}",
        *card.positive_examples,
    ]

    result: list[str] = []

    for text in texts:
        normalized = str(text).strip()
        if normalized:
            result.append(normalized)

    return result


def _build_topic_records(
    encoder: BaseEncoder,
    topic_cards: list[TopicCard],
) -> list[TopicEmbeddingRecord]:
    records: list[TopicEmbeddingRecord] = []

    for card in topic_cards:
        positive_texts = _build_positive_texts_for_topic(card)

        positive_vectors = encoder.encode(positive_texts)
        positive_vectors = l2_normalize(positive_vectors)

        negative_vectors: np.ndarray | None = None
        if card.negative_examples:
            negative_vectors = encoder.encode(card.negative_examples)
            negative_vectors = l2_normalize(negative_vectors)

        records.append(
            TopicEmbeddingRecord(
                group_id=card.group_id,
                group_name=card.group_name,
                topic_id=card.topic_id,
                topic_name=card.topic_name,
                status=card.status,
                positive_vectors=positive_vectors,
                negative_vectors=negative_vectors,
            )
        )

    return records