from __future__ import annotations

from dataclasses import dataclass

from src.semantic_router.taxonomy.schema import Group, Taxonomy, Topic


@dataclass(frozen=True)
class TopicCard:
    group_id: str
    group_name: str
    group_description: str
    topic_id: str
    topic_name: str
    topic_description: str
    status: str
    card_text: str
    positive_examples: list[str]
    negative_examples: list[str]


@dataclass(frozen=True)
class GroupCard:
    group_id: str
    group_name: str
    card_text: str
    topic_ids: list[str]


def build_topic_cards(
    taxonomy: Taxonomy,
    allowed_statuses: set[str] | None = None,
) -> list[TopicCard]:
    cards: list[TopicCard] = []

    for group in taxonomy.groups:
        for topic in group.topics:
            if allowed_statuses is not None and topic.status not in allowed_statuses:
                continue

            cards.append(_build_topic_card(group, topic))

    return cards


def build_group_cards(
    taxonomy: Taxonomy,
    allowed_statuses: set[str] | None = None,
) -> list[GroupCard]:
    cards: list[GroupCard] = []

    for group in taxonomy.groups:
        topic_names: list[str] = []
        topic_ids: list[str] = []

        for topic in group.topics:
            if allowed_statuses is not None and topic.status not in allowed_statuses:
                continue

            topic_names.append(topic.topic_name)
            topic_ids.append(topic.topic_id)

        card_text = "\n".join(
            [
                f"Группа: {group.group_name}",
                f"Описание группы: {group.group_description}",
                "Темы группы:",
                *[f"- {name}" for name in topic_names],
            ]
        )

        cards.append(
            GroupCard(
                group_id=group.group_id,
                group_name=group.group_name,
                card_text=card_text,
                topic_ids=topic_ids,
            )
        )

    return cards

def _build_topic_card(group: Group, topic: Topic) -> TopicCard:
    positive_block = "\n".join(
        f"- {example}"
        for example in topic.positive_examples
        if example and example.strip()
    )

    negative_block = "\n".join(
        f"- {example}"
        for example in topic.negative_examples
        if example and example.strip()
    )

    card_parts = [
        f"Группа: {group.group_name}",
        f"Описание группы: {group.group_description}",
        f"Тема: {topic.topic_name}",
        f"Описание темы: {topic.topic_description}",
    ]

    if positive_block:
        card_parts.extend(
            [
                "Эта тема подходит для обращений:",
                positive_block,
            ]
        )

    if negative_block:
        card_parts.extend(
            [
                "Эта тема НЕ подходит для обращений:",
                negative_block,
            ]
        )

    card_text = "\n".join(card_parts)

    return TopicCard(
        group_id=group.group_id,
        group_name=group.group_name,
        group_description=group.group_description,
        topic_id=topic.topic_id,
        topic_name=topic.topic_name,
        topic_description=topic.topic_description,
        status=topic.status,
        card_text=card_text,
        positive_examples=topic.positive_examples,
        negative_examples=topic.negative_examples,
    )