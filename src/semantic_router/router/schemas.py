from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoreItem:
    id: str
    name: str
    score: float
    group_id: str | None = None
    group_name: str | None = None


@dataclass(frozen=True)
class RoutePrediction:
    text: str
    normalized_text: str

    group_id: str
    group_name: str

    topic_id: str
    topic_name: str

    is_unknown: bool
    confidence: float
    margin: float
    decision_reason: str

    top_groups: list[ScoreItem]
    top_topics: list[ScoreItem]

    def to_dict(self) -> dict:
        return asdict(self)