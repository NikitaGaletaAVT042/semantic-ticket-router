from __future__ import annotations

from src.semantic_router.taxonomy.card_builder import build_group_cards, build_topic_cards
from src.semantic_router.taxonomy.schema import Group, Taxonomy, Topic, UnknownTopic


def test_build_topic_cards():
    taxonomy = Taxonomy(
        taxonomy_version="test",
        language="ru",
        description="test",
        groups=[
            Group(
                group_id="g1",
                group_name="Group 1",
                group_description="Group description",
                unknown_topic=UnknownTopic("g1.unknown", "UNKNOWN: Group 1"),
                topics=[
                    Topic(
                        topic_id="g1.t1",
                        topic_name="Topic 1",
                        topic_description="Topic description",
                        positive_examples=["positive"],
                        negative_examples=["negative"],
                        status="active",
                    )
                ],
            )
        ],
    )

    cards = build_topic_cards(taxonomy, allowed_statuses={"active"})

    assert len(cards) == 1
    assert cards[0].topic_id == "g1.t1"
    assert "Topic 1" in cards[0].card_text
    assert "positive" in cards[0].card_text


def test_build_group_cards():
    taxonomy = Taxonomy(
        taxonomy_version="test",
        language="ru",
        description="test",
        groups=[
            Group(
                group_id="g1",
                group_name="Group 1",
                group_description="Group description",
                unknown_topic=UnknownTopic("g1.unknown", "UNKNOWN: Group 1"),
                topics=[
                    Topic(
                        topic_id="g1.t1",
                        topic_name="Topic 1",
                        topic_description="Topic description",
                        positive_examples=["positive"],
                        negative_examples=["negative"],
                        status="active",
                    )
                ],
            )
        ],
    )

    cards = build_group_cards(taxonomy, allowed_statuses={"active"})

    assert len(cards) == 1
    assert cards[0].group_id == "g1"
    assert "Topic 1" in cards[0].card_text