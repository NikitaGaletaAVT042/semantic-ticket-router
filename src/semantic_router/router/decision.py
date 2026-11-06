from __future__ import annotations

from dataclasses import dataclass

from src.semantic_router.router.config import RoutingConfig
from src.semantic_router.router.schemas import ScoreItem
from src.semantic_router.taxonomy.schema import Group, Taxonomy, Topic


@dataclass(frozen=True)
class Decision:
    group_id: str
    group_name: str
    topic_id: str
    topic_name: str
    is_unknown: bool
    confidence: float
    margin: float
    reason: str


def make_decision(
    *,
    taxonomy: Taxonomy,
    group_scores: list[ScoreItem],
    topic_scores: list[ScoreItem],
    config: RoutingConfig,
) -> Decision:
    """
    Topic-first decision policy.

    Почему topic-first:
    - в текущей системе topic_scores часто уже содержат правильную тему в top-1/top-3;
    - group_scores слабее и не должны блокировать сильный topic match;
    - group unknown нужен только как fallback, когда topic-level недостаточно уверен.
    """
    if not topic_scores and not group_scores:
        return _global_unknown("no_scores")

    topic_by_id = taxonomy.topic_by_id()
    group_by_id = taxonomy.group_by_id()

    best_topic = topic_scores[0] if topic_scores else None
    second_topic_score = topic_scores[1].score if len(topic_scores) > 1 else 0.0
    topic_margin = best_topic.score - second_topic_score if best_topic else 0.0

    if best_topic is not None:
        topic = topic_by_id.get(best_topic.id)
        group_id = best_topic.group_id or _infer_group_id_from_topic_id(best_topic.id)
        group = group_by_id.get(group_id)

        if topic is not None and group is not None:
            topic_is_strong = best_topic.score >= config.topic_min_similarity
            margin_is_ok = (
                not config.reject_on_low_margin
                or topic_margin >= config.topic_min_margin
            )
            topic_is_high_confidence = (
                best_topic.score >= config.topic_high_confidence
                and topic_margin >= config.topic_high_confidence_margin
            )

            if topic_is_strong and (margin_is_ok or topic_is_high_confidence):
                return Decision(
                    group_id=group.group_id,
                    group_name=group.group_name,
                    topic_id=topic.topic_id,
                    topic_name=topic.topic_name,
                    is_unknown=False,
                    confidence=float(best_topic.score),
                    margin=float(topic_margin),
                    reason="matched_topic_first",
                )

            if topic_is_strong and not margin_is_ok:
                return _group_unknown(
                    group=group,
                    confidence=float(best_topic.score),
                    margin=float(topic_margin),
                    reason="low_topic_margin",
                )

    best_group = group_scores[0] if group_scores else None

    if best_group is None:
        return _global_unknown("no_group_scores")

    group = group_by_id.get(best_group.id)

    if group is None:
        return _global_unknown("best_group_not_found_in_taxonomy")

    if best_group.score >= config.group_min_similarity:
        return _group_unknown(
            group=group,
            confidence=float(best_group.score),
            margin=0.0,
            reason="low_topic_similarity_group_known",
        )

    return _global_unknown("low_topic_similarity_and_low_group_similarity")


def _group_unknown(
    *,
    group: Group,
    confidence: float,
    margin: float,
    reason: str,
) -> Decision:
    return Decision(
        group_id=group.group_id,
        group_name=group.group_name,
        topic_id=group.unknown_topic.topic_id,
        topic_name=group.unknown_topic.topic_name,
        is_unknown=True,
        confidence=float(confidence),
        margin=float(margin),
        reason=reason,
    )


def _global_unknown(reason: str) -> Decision:
    return Decision(
        group_id="unknown",
        group_name="UNKNOWN",
        topic_id="unknown",
        topic_name="UNKNOWN",
        is_unknown=True,
        confidence=0.0,
        margin=0.0,
        reason=reason,
    )


def _infer_group_id_from_topic_id(topic_id: str) -> str:
    if "." not in topic_id:
        return ""

    return topic_id.split(".", 1)[0]