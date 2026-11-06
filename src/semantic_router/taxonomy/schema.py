from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_TOPIC_STATUSES = {
    "active",
    "candidate_needs_review",
    "not_found_in_current_dataset",
    "disabled",
}


@dataclass(frozen=True)
class UnknownTopic:
    topic_id: str
    topic_name: str
    is_unknown: bool = True


@dataclass(frozen=True)
class Topic:
    topic_id: str
    topic_name: str
    topic_description: str
    positive_examples: list[str]
    negative_examples: list[str]
    status: str = "active"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def group_id(self) -> str:
        if "." not in self.topic_id:
            return ""
        return self.topic_id.split(".", 1)[0]


@dataclass(frozen=True)
class Group:
    group_id: str
    group_name: str
    group_description: str
    unknown_topic: UnknownTopic
    topics: list[Topic]


@dataclass(frozen=True)
class Taxonomy:
    taxonomy_version: str
    language: str
    description: str
    groups: list[Group]

    def group_by_id(self) -> dict[str, Group]:
        return {group.group_id: group for group in self.groups}

    def topic_by_id(self) -> dict[str, Topic]:
        result: dict[str, Topic] = {}
        for group in self.groups:
            for topic in group.topics:
                result[topic.topic_id] = topic
        return result