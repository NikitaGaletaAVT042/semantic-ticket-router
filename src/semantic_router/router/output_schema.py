from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.semantic_router.router.schemas import RoutePrediction, ScoreItem


@dataclass(frozen=True)
class PredictionInputMeta:
    text: str
    normalized_text: str
    text_chars: int
    chunks_count: int


@dataclass(frozen=True)
class PredictionSourceMeta:
    row_index: int | None = None
    key: str | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class PredictionDebugMeta:
    encoder_model: str
    calibration_applied: bool
    routing_config: dict[str, Any]


@dataclass(frozen=True)
class PredictionOutput:
    schema_version: str
    source: PredictionSourceMeta
    input: PredictionInputMeta
    prediction: dict[str, Any]
    scores: dict[str, list[dict[str, Any]]]
    debug: PredictionDebugMeta

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_item_to_dict(item: ScoreItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "score": float(item.score),
        "group_id": item.group_id,
        "group_name": item.group_name,
    }


def build_prediction_output(
    *,
    route_prediction: RoutePrediction,
    source: PredictionSourceMeta,
    chunks_count: int,
    encoder_model: str,
    calibration_applied: bool,
    routing_config: dict[str, Any],
) -> PredictionOutput:
    return PredictionOutput(
        schema_version="semantic_router_prediction_v1",
        source=source,
        input=PredictionInputMeta(
            text=route_prediction.text,
            normalized_text=route_prediction.normalized_text,
            text_chars=len(route_prediction.normalized_text),
            chunks_count=chunks_count,
        ),
        prediction={
            "group_id": route_prediction.group_id,
            "group_name": route_prediction.group_name,
            "topic_id": route_prediction.topic_id,
            "topic_name": route_prediction.topic_name,
            "is_unknown": route_prediction.is_unknown,
            "confidence": float(route_prediction.confidence),
            "margin": float(route_prediction.margin),
            "decision_reason": route_prediction.decision_reason,
        },
        scores={
            "top_groups": [
                score_item_to_dict(item)
                for item in route_prediction.top_groups
            ],
            "top_topics": [
                score_item_to_dict(item)
                for item in route_prediction.top_topics
            ],
        },
        debug=PredictionDebugMeta(
            encoder_model=encoder_model,
            calibration_applied=calibration_applied,
            routing_config=routing_config,
        ),
    )