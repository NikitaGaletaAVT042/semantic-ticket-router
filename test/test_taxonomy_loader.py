from __future__ import annotations

import json

import pytest

from src.semantic_router.taxonomy.loader import TaxonomyValidationError, load_taxonomy


def test_load_valid_taxonomy(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(
        json.dumps(
            {
                "taxonomy_version": "test",
                "language": "ru",
                "description": "test",
                "groups": [
                    {
                        "group_id": "g1",
                        "group_name": "Group 1",
                        "group_description": "Description",
                        "unknown_topic": {
                            "topic_id": "g1.unknown",
                            "topic_name": "UNKNOWN: Group 1",
                            "is_unknown": True,
                        },
                        "topics": [
                            {
                                "topic_id": "g1.t1",
                                "topic_name": "Topic 1",
                                "topic_description": "Topic description",
                                "positive_examples": ["example"],
                                "negative_examples": ["negative"],
                                "status": "active",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    taxonomy = load_taxonomy(path)

    assert taxonomy.taxonomy_version == "test"
    assert taxonomy.groups[0].group_id == "g1"
    assert taxonomy.groups[0].topics[0].topic_id == "g1.t1"


def test_active_topic_without_positive_examples_is_invalid(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(
        json.dumps(
            {
                "taxonomy_version": "test",
                "language": "ru",
                "description": "test",
                "groups": [
                    {
                        "group_id": "g1",
                        "group_name": "Group 1",
                        "group_description": "Description",
                        "unknown_topic": {
                            "topic_id": "g1.unknown",
                            "topic_name": "UNKNOWN: Group 1",
                            "is_unknown": True,
                        },
                        "topics": [
                            {
                                "topic_id": "g1.t1",
                                "topic_name": "Topic 1",
                                "topic_description": "Topic description",
                                "positive_examples": [],
                                "negative_examples": [],
                                "status": "active",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaxonomyValidationError):
        load_taxonomy(path)