from __future__ import annotations

from src.semantic_router.router.config import RoutingConfig
from src.semantic_router.router.decision import make_decision
from src.semantic_router.router.schemas import ScoreItem
from src.semantic_router.taxonomy.schema import Group, Taxonomy, Topic, UnknownTopic


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        taxonomy_version="test",
        language="ru",
        description="test",
        groups=[
            Group(
                group_id="g1",
                group_name="Group 1",
                group_description="Group 1 description",
                unknown_topic=UnknownTopic("g1.unknown", "UNKNOWN: Group 1"),
                topics=[
                    Topic(
                        topic_id="g1.t1",
                        topic_name="Topic 1",
                        topic_description="Topic 1 description",
                        positive_examples=["positive"],
                        negative_examples=[],
                        status="active",
                    ),
                    Topic(
                        topic_id="g1.t2",
                        topic_name="Topic 2",
                        topic_description="Topic 2 description",
                        positive_examples=["positive"],
                        negative_examples=[],
                        status="active",
                    ),
                ],
            ),
            Group(
                group_id="g2",
                group_name="Group 2",
                group_description="Group 2 description",
                unknown_topic=UnknownTopic("g2.unknown", "UNKNOWN: Group 2"),
                topics=[
                    Topic(
                        topic_id="g2.t1",
                        topic_name="Topic 3",
                        topic_description="Topic 3 description",
                        positive_examples=["positive"],
                        negative_examples=[],
                        status="active",
                    )
                ],
            ),
        ],
    )


def _config() -> RoutingConfig:
    return RoutingConfig(
        group_min_similarity=0.2,
        group_min_margin=0.0,
        topic_min_similarity=0.3,
        topic_min_margin=0.04,
        reject_on_low_margin=True,
        topic_high_confidence=0.55,
        topic_high_confidence_margin=0.08,
    )


def test_topic_first_returns_best_topic_even_if_best_group_is_different():
    decision = make_decision(
        taxonomy=_taxonomy(),
        group_scores=[
            ScoreItem("g2", "Group 2", 0.70, group_id="g2", group_name="Group 2"),
            ScoreItem("g1", "Group 1", 0.40, group_id="g1", group_name="Group 1"),
        ],
        topic_scores=[
            ScoreItem("g1.t1", "Topic 1", 0.60, group_id="g1", group_name="Group 1"),
            ScoreItem("g2.t1", "Topic 3", 0.40, group_id="g2", group_name="Group 2"),
        ],
        config=_config(),
    )

    assert decision.is_unknown is False
    assert decision.group_id == "g1"
    assert decision.topic_id == "g1.t1"
    assert decision.reason == "matched_topic_first"


def test_returns_group_unknown_when_best_topic_has_low_margin():
    decision = make_decision(
        taxonomy=_taxonomy(),
        group_scores=[
            ScoreItem("g1", "Group 1", 0.50, group_id="g1", group_name="Group 1"),
        ],
        topic_scores=[
            ScoreItem("g1.t1", "Topic 1", 0.50, group_id="g1", group_name="Group 1"),
            ScoreItem("g1.t2", "Topic 2", 0.49, group_id="g1", group_name="Group 1"),
        ],
        config=_config(),
    )

    assert decision.is_unknown is True
    assert decision.group_id == "g1"
    assert decision.topic_id == "g1.unknown"
    assert decision.reason == "low_topic_margin"


def test_high_confidence_topic_can_pass_even_with_low_margin_policy():
    decision = make_decision(
        taxonomy=_taxonomy(),
        group_scores=[
            ScoreItem("g1", "Group 1", 0.50, group_id="g1", group_name="Group 1"),
        ],
        topic_scores=[
            ScoreItem("g1.t1", "Topic 1", 0.70, group_id="g1", group_name="Group 1"),
            ScoreItem("g1.t2", "Topic 2", 0.64, group_id="g1", group_name="Group 1"),
        ],
        config=RoutingConfig(
            group_min_similarity=0.2,
            topic_min_similarity=0.3,
            topic_min_margin=0.1,
            reject_on_low_margin=True,
            topic_high_confidence=0.65,
            topic_high_confidence_margin=0.05,
        ),
    )

    assert decision.is_unknown is False
    assert decision.topic_id == "g1.t1"
    assert decision.reason == "matched_topic_first"


def test_returns_group_unknown_when_topic_is_weak_but_group_is_known():
    decision = make_decision(
        taxonomy=_taxonomy(),
        group_scores=[
            ScoreItem("g1", "Group 1", 0.50, group_id="g1", group_name="Group 1"),
        ],
        topic_scores=[
            ScoreItem("g1.t1", "Topic 1", 0.20, group_id="g1", group_name="Group 1"),
        ],
        config=_config(),
    )

    assert decision.is_unknown is True
    assert decision.group_id == "g1"
    assert decision.topic_id == "g1.unknown"
    assert decision.reason == "low_topic_similarity_group_known"


def test_returns_global_unknown_when_topic_and_group_are_weak():
    decision = make_decision(
        taxonomy=_taxonomy(),
        group_scores=[
            ScoreItem("g1", "Group 1", 0.10, group_id="g1", group_name="Group 1"),
        ],
        topic_scores=[
            ScoreItem("g1.t1", "Topic 1", 0.20, group_id="g1", group_name="Group 1"),
        ],
        config=_config(),
    )

    assert decision.is_unknown is True
    assert decision.group_id == "unknown"
    assert decision.topic_id == "unknown"
    assert decision.reason == "low_topic_similarity_and_low_group_similarity"


def test_returns_global_unknown_when_no_scores():
    decision = make_decision(
        taxonomy=_taxonomy(),
        group_scores=[],
        topic_scores=[],
        config=_config(),
    )

    assert decision.is_unknown is True
    assert decision.group_id == "unknown"
    assert decision.topic_id == "unknown"
    assert decision.reason == "no_scores"