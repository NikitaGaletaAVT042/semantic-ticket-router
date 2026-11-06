from __future__ import annotations

import json

from scripts.evaluate_against_markup import (
    build_eval_items,
    build_prediction_by_key,
    read_markup,
)
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
                group_description="Description",
                unknown_topic=UnknownTopic(
                    topic_id="g1.unknown",
                    topic_name="UNKNOWN: Group 1",
                    is_unknown=True,
                ),
                topics=[
                    Topic(
                        topic_id="g1.t1",
                        topic_name="Topic 1",
                        topic_description="Topic description",
                        positive_examples=["example"],
                        negative_examples=[],
                        status="active",
                    )
                ],
            )
        ],
    )


def test_read_markup(tmp_path):
    markup_path = tmp_path / "markup.txt"
    markup_path.write_text(
        "\n".join(
            [
                "Topic 1",
                "https://jira.technokad.ru/browse/TXSUP-1",
                "https://jira.technokad.ru/browse/TXSUP-2",
            ]
        ),
        encoding="utf-8",
    )

    result = read_markup(markup_path)

    assert result == {
        "TXSUP-1": "Topic 1",
        "TXSUP-2": "Topic 1",
    }


def test_build_prediction_by_key():
    rows = [
        {
            "key": "TXSUP-1",
            "prediction": {
                "topic_id": "g1.t1",
            },
        }
    ]

    result = build_prediction_by_key(rows)

    assert "TXSUP-1" in result
    assert result["TXSUP-1"]["prediction"]["topic_id"] == "g1.t1"


def test_build_eval_items_maps_expected_label_to_topic_id():
    expected_by_key = {
        "TXSUP-1": "Topic 1",
    }
    predictions_by_key = {
        "TXSUP-1": {
            "key": "TXSUP-1",
            "prediction": {
                "group_id": "g1",
                "topic_id": "g1.t1",
                "is_unknown": False,
                "top_topics": [
                    {
                        "id": "g1.t1",
                        "score": 0.8,
                    }
                ],
            },
        }
    }

    items, missing = build_eval_items(
        expected_by_key=expected_by_key,
        predictions_by_key=predictions_by_key,
        taxonomy=_taxonomy(),
    )

    assert missing == []
    assert len(items) == 1
    assert items[0].expected_topic_id == "g1.t1"
    assert items[0].predicted_topic_id == "g1.t1"
    assert items[0].expected_group_id == "g1"
    assert items[0].predicted_group_id == "g1"

def test_build_eval_items_keeps_confidence_margin_and_reason():
    expected_by_key = {
        "TXSUP-1": "Topic 1",
    }
    predictions_by_key = {
        "TXSUP-1": {
            "key": "TXSUP-1",
            "prediction": {
                "group_id": "g1",
                "topic_id": "g1.t1",
                "is_unknown": False,
                "confidence": 0.77,
                "margin": 0.12,
                "decision_reason": "matched_topic_first",
                "top_topics": [
                    {
                        "id": "g1.t1",
                        "score": 0.77,
                    }
                ],
            },
        }
    }

    items, missing = build_eval_items(
        expected_by_key=expected_by_key,
        predictions_by_key=predictions_by_key,
        taxonomy=_taxonomy(),
    )

    assert missing == []
    assert items[0].confidence == 0.77
    assert items[0].margin == 0.12
    assert items[0].decision_reason == "matched_topic_first"