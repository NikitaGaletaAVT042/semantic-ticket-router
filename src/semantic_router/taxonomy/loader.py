from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.semantic_router.taxonomy.schema import (
    ALLOWED_TOPIC_STATUSES,
    Group,
    Taxonomy,
    Topic,
    UnknownTopic,
)


class TaxonomyValidationError(ValueError):
    pass


def load_taxonomy(path: str | Path) -> Taxonomy:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    taxonomy = _parse_taxonomy(raw)
    validate_taxonomy(taxonomy)
    return taxonomy


def _parse_taxonomy(raw: dict[str, Any]) -> Taxonomy:
    groups: list[Group] = []

    for raw_group in raw.get("groups", []):
        raw_unknown = raw_group.get("unknown_topic", {})

        unknown_topic = UnknownTopic(
            topic_id=str(raw_unknown.get("topic_id", "")).strip(),
            topic_name=str(raw_unknown.get("topic_name", "")).strip(),
            is_unknown=bool(raw_unknown.get("is_unknown", True)),
        )

        topics: list[Topic] = []
        for raw_topic in raw_group.get("topics", []):
            known_keys = {
                "topic_id",
                "topic_name",
                "topic_description",
                "positive_examples",
                "negative_examples",
                "status",
            }

            extra = {
                key: value
                for key, value in raw_topic.items()
                if key not in known_keys
            }

            topic = Topic(
                topic_id=str(raw_topic.get("topic_id", "")).strip(),
                topic_name=str(raw_topic.get("topic_name", "")).strip(),
                topic_description=str(raw_topic.get("topic_description", "")).strip(),
                positive_examples=_clean_string_list(raw_topic.get("positive_examples", [])),
                negative_examples=_clean_string_list(raw_topic.get("negative_examples", [])),
                status=str(raw_topic.get("status", "active")).strip(),
                extra=extra,
            )
            topics.append(topic)

        group = Group(
            group_id=str(raw_group.get("group_id", "")).strip(),
            group_name=str(raw_group.get("group_name", "")).strip(),
            group_description=str(raw_group.get("group_description", "")).strip(),
            unknown_topic=unknown_topic,
            topics=topics,
        )
        groups.append(group)

    return Taxonomy(
        taxonomy_version=str(raw.get("taxonomy_version", "")).strip(),
        language=str(raw.get("language", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        groups=groups,
    )


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)

    return cleaned


def validate_taxonomy(taxonomy: Taxonomy) -> None:
    errors: list[str] = []

    if not taxonomy.taxonomy_version:
        errors.append("taxonomy_version is empty")

    if not taxonomy.language:
        errors.append("language is empty")

    if not taxonomy.groups:
        errors.append("groups is empty")

    group_ids: set[str] = set()
    topic_ids: set[str] = set()

    for group in taxonomy.groups:
        if not group.group_id:
            errors.append("group_id is empty")

        if group.group_id in group_ids:
            errors.append(f"duplicated group_id: {group.group_id}")
        group_ids.add(group.group_id)

        if not group.group_name:
            errors.append(f"group_name is empty for group_id={group.group_id}")

        expected_unknown_id = f"{group.group_id}.unknown"
        if group.unknown_topic.topic_id != expected_unknown_id:
            errors.append(
                f"unknown_topic.topic_id must be {expected_unknown_id}, "
                f"got {group.unknown_topic.topic_id}"
            )

        if not group.unknown_topic.is_unknown:
            errors.append(f"unknown_topic.is_unknown must be true for {group.group_id}")

        if not group.topics:
            errors.append(f"group has no topics: {group.group_id}")

        for topic in group.topics:
            if not topic.topic_id:
                errors.append(f"topic_id is empty in group {group.group_id}")

            if topic.topic_id in topic_ids:
                errors.append(f"duplicated topic_id: {topic.topic_id}")
            topic_ids.add(topic.topic_id)

            if not topic.topic_id.startswith(f"{group.group_id}."):
                errors.append(
                    f"topic_id={topic.topic_id} does not belong to group_id={group.group_id}"
                )

            if not topic.topic_name:
                errors.append(f"topic_name is empty for topic_id={topic.topic_id}")

            if not topic.topic_description:
                errors.append(f"topic_description is empty for topic_id={topic.topic_id}")

            if topic.status not in ALLOWED_TOPIC_STATUSES:
                errors.append(
                    f"invalid status={topic.status} for topic_id={topic.topic_id}"
                )

            if topic.status == "active" and not topic.positive_examples:
                errors.append(
                    f"active topic has empty positive_examples: {topic.topic_id}"
                )

    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise TaxonomyValidationError(f"Invalid taxonomy:\n{joined}")