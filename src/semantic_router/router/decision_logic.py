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


def find_group(taxonomy: Taxonomy, group_id: str) -> Group | None:
    for group in taxonomy.groups:
        if group.group_id == group_id:
            return group

    return None


def find_topic(taxonomy: Taxonomy, topic_id: str) -> Topic | None:
    for group in taxonomy.groups:
        for topic in group.topics:
            if topic.topic_id == topic_id:
                return topic

    return None


def make_decision(
    *,
    taxonomy: Taxonomy,
    group_scores: list[ScoreItem],
    topic_scores: list[ScoreItem],
    config: RoutingConfig,
) -> Decision:
    if not group_scores:
        return _global_unknown("no_group_scores")

    best_group = group_scores[0]

    group = find_group(taxonomy, best_group.id)
    if group is None:
        return _global_unknown("best_group_not_found_in_taxonomy")

    if best_group.score < config.group_min_similarity:
        return _group_unknown(
            group_id=group.group_id,
            group_name=group.group_name,
            topic_id=group.unknown_topic.topic_id,
            topic_name=group.unknown_topic.topic_name,
            confidence=best_group.score,
            margin=0.0,
            reason="low_group_similarity",
        )

    group_topic_scores = [
        item for item in topic_scores
        if item.group_id == group.group_id
    ]

    if not group_topic_scores:
        return _group_unknown(
            group_id=group.group_id,
            group_name=group.group_name,
            topic_id=group.unknown_topic.topic_id,
            topic_name=group.unknown_topic.topic_name,
            confidence=best_group.score,
            margin=0.0,
            reason="no_topic_scores_for_group",
        )

    best_topic = group_topic_scores[0]
    second_score = group_topic_scores[1].score if len(group_topic_scores) > 1 else 0.0
    margin = best_topic.score - second_score

    if best_topic.score < config.topic_min_similarity:
        return _group_unknown(
            group_id=group.group_id,
            group_name=group.group_name,
            topic_id=group.unknown_topic.topic_id,
            topic_name=group.unknown_topic.topic_name,
            confidence=best_topic.score,
            margin=margin,
            reason="low_topic_similarity",
        )

    if config.reject_on_low_margin and margin < config.topic_min_margin:
        return _group_unknown(
            group_id=group.group_id,
            group_name=group.group_name,
            topic_id=group.unknown_topic.topic_id,
            topic_name=group.unknown_topic.topic_name,
            confidence=best_topic.score,
            margin=margin,
            reason="ambiguous_low_topic_margin",
        )

    topic = find_topic(taxonomy, best_topic.id)
    if topic is None:
        return _group_unknown(
            group_id=group.group_id,
            group_name=group.group_name,
            topic_id=group.unknown_topic.topic_id,
            topic_name=group.unknown_topic.topic_name,
            confidence=best_topic.score,
            margin=margin,
            reason="best_topic_not_found_in_taxonomy",
        )

    return Decision(
        group_id=group.group_id,
        group_name=group.group_name,
        topic_id=topic.topic_id,
        topic_name=topic.topic_name,
        is_unknown=False,
        confidence=best_topic.score,
        margin=margin,
        reason="matched",
    )


def _group_unknown(
    *,
    group_id: str,
    group_name: str,
    topic_id: str,
    topic_name: str,
    confidence: float,
    margin: float,
    reason: str,
) -> Decision:
    return Decision(
        group_id=group_id,
        group_name=group_name,
        topic_id=topic_id,
        topic_name=topic_name,
        is_unknown=True,
        confidence=confidence,
        margin=margin,
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