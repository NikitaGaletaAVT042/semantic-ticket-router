from __future__ import annotations

import numpy as np

from src.semantic_router.embeddings.embedding_index import (
    EmbeddingIndex,
    TopicEmbeddingRecord,
    build_embedding_index,
)
from src.semantic_router.embeddings.encoder import (
    BaseEncoder,
    HashingEncoder,
    SentenceTransformerEncoder,
)
from src.semantic_router.embeddings.vector_match import cosine_similarity_matrix
from src.semantic_router.preprocessing.normalizer import normalize_text
from src.semantic_router.preprocessing.chunker import split_text_for_routing
from src.semantic_router.router.config import RouterConfig
from src.semantic_router.router.decision import make_decision
from src.semantic_router.router.schemas import RoutePrediction, ScoreItem
from src.semantic_router.taxonomy.card_builder import build_group_cards, build_topic_cards
from src.semantic_router.taxonomy.loader import load_taxonomy
from src.semantic_router.taxonomy.schema import Taxonomy
from src.semantic_router.topic_calibration import load_topic_calibration

class SemanticTaxonomyRouter:
    def __init__(
        self,
        *,
        config: RouterConfig,
        taxonomy: Taxonomy,
        encoder: BaseEncoder,
        index: EmbeddingIndex,
    ) -> None:
        self.config = config
        self.taxonomy = taxonomy
        self.encoder = encoder
        self.index = index
        self.topic_calibration = load_topic_calibration()

    @classmethod
    def from_config(cls, config: RouterConfig) -> "SemanticTaxonomyRouter":
        taxonomy = load_taxonomy(config.taxonomy_path)

        encoder: BaseEncoder
        if config.embedding.use_hashing_encoder:
            encoder = HashingEncoder(dim=config.embedding.hashing_dim)
        else:
            encoder = SentenceTransformerEncoder(
                model_name=config.embedding.model_name,
                batch_size=config.embedding.batch_size,
            )

        allowed_statuses = set(config.routing.active_statuses)

        topic_cards = build_topic_cards(
            taxonomy,
            allowed_statuses=allowed_statuses,
        )
        group_cards = build_group_cards(
            taxonomy,
            allowed_statuses=allowed_statuses,
        )

        index = build_embedding_index(
            encoder=encoder,
            topic_cards=topic_cards,
            group_cards=group_cards,
        )

        return cls(
            config=config,
            taxonomy=taxonomy,
            encoder=encoder,
            index=index,
        )

    def predict(self, text: str) -> RoutePrediction:
        normalized_text = normalize_text(text)

        texts_to_encode = [normalized_text]
        chunks: list[str] = []

        if self.config.routing.use_chunk_scoring:
            chunks = split_text_for_routing(
                normalized_text,
                min_chunk_chars=self.config.routing.min_chunk_chars,
                max_chunk_chars=self.config.routing.max_chunk_chars,
                max_chunks=self.config.routing.max_chunks,
            )
            texts_to_encode.extend(chunks)

        encoded_vectors = self.encoder.encode(texts_to_encode)
        query_vector = encoded_vectors[0]

        chunk_vectors: np.ndarray | None = None
        if len(encoded_vectors) > 1:
            chunk_vectors = encoded_vectors[1:]

        group_scores = self._score_groups(query_vector)
        group_score_by_id = {item.id: item.score for item in group_scores}

        topic_scores = self._score_topics(
            query_vector,
            self.index.topics,
            group_score_by_id=group_score_by_id,
            chunk_vectors=chunk_vectors,
        )
        topic_scores = self._apply_topic_calibration_to_scores(topic_scores)
        decision = make_decision(
            taxonomy=self.taxonomy,
            group_scores=group_scores,
            topic_scores=topic_scores,
            config=self.config.routing,
        )

        return RoutePrediction(
            text=text,
            normalized_text=normalized_text,
            group_id=decision.group_id,
            group_name=decision.group_name,
            topic_id=decision.topic_id,
            topic_name=decision.topic_name,
            is_unknown=decision.is_unknown,
            confidence=float(decision.confidence),
            margin=float(decision.margin),
            decision_reason=decision.reason,
            top_groups=group_scores[: self.config.routing.return_top_k_groups],
            top_topics=topic_scores[: self.config.routing.return_top_k_topics],
        )

    def predict_many(self, texts: list[str]) -> list[RoutePrediction]:
        return [self.predict(text) for text in texts]

    def _apply_topic_calibration_to_scores(self, topic_scores: list[ScoreItem]) -> list[ScoreItem]:
        if not self.topic_calibration:
            return topic_scores

        calibrated_scores: list[ScoreItem] = []

        for item in topic_scores:
            bias = self.topic_calibration.get(
                item.id,
                self.topic_calibration.get(item.name, 0.0),
            )

            if bias == 0.0:
                calibrated_scores.append(item)
                continue

            calibrated_scores.append(
                ScoreItem(
                    id=item.id,
                    name=item.name,
                    score=float(item.score) + float(bias),
                    group_id=item.group_id,
                    group_name=item.group_name,
                )
            )

        return sorted(calibrated_scores, key=lambda item: item.score, reverse=True)

    def _score_groups(self, query_vector: np.ndarray) -> list[ScoreItem]:
        if not self.index.groups:
            return []

        matrix = np.vstack([record.vector for record in self.index.groups])
        scores = cosine_similarity_matrix(query_vector, matrix)

        result = [
            ScoreItem(
                id=record.group_id,
                name=record.group_name,
                score=float(score),
                group_id=record.group_id,
                group_name=record.group_name,
            )
            for record, score in zip(self.index.groups, scores)
        ]

        return sorted(result, key=lambda item: item.score, reverse=True)

    def _score_topics(
            self,
            query_vector: np.ndarray,
            topic_records: list[TopicEmbeddingRecord],
            *,
            group_score_by_id: dict[str, float] | None = None,
            chunk_vectors: np.ndarray | None = None,
    ) -> list[ScoreItem]:
        if not topic_records:
            return []

        result: list[ScoreItem] = []

        for record in topic_records:
            full_score = self._score_vector_against_topic(
                query_vector=query_vector,
                record=record,
            )

            score = full_score

            if chunk_vectors is not None and len(chunk_vectors) > 0:
                chunk_scores = [
                    self._score_vector_against_topic(
                        query_vector=chunk_vector,
                        record=record,
                    )
                    for chunk_vector in chunk_vectors
                ]

                best_chunk_score = max(chunk_scores) if chunk_scores else full_score

                score = (
                        (1.0 - self.config.routing.chunk_score_weight) * full_score
                        + self.config.routing.chunk_score_weight * best_chunk_score
                )

            if record.negative_vectors is not None:
                negative_scores = cosine_similarity_matrix(
                    query_vector,
                    record.negative_vectors,
                )
                max_negative_score = float(np.max(negative_scores))
                score -= self.config.routing.negative_penalty * max(
                    0.0,
                    max_negative_score,
                )

            group_prior = 0.0
            if group_score_by_id is not None:
                group_prior = group_score_by_id.get(record.group_id, 0.0)

            score += self.config.routing.group_prior_weight * group_prior

            result.append(
                ScoreItem(
                    id=record.topic_id,
                    name=record.topic_name,
                    score=float(score),
                    group_id=record.group_id,
                    group_name=record.group_name,
                )
            )

        return sorted(result, key=lambda item: item.score, reverse=True)

    def _score_vector_against_topic(
            self,
            *,
            query_vector: np.ndarray,
            record: TopicEmbeddingRecord,
    ) -> float:
        positive_scores = cosine_similarity_matrix(
            query_vector,
            record.positive_vectors,
        )

        top_k = min(
            self.config.routing.topic_score_top_k,
            len(positive_scores),
        )

        if top_k <= 0:
            return 0.0

        top_positive_scores = np.sort(positive_scores)[-top_k:]
        return float(np.mean(top_positive_scores))