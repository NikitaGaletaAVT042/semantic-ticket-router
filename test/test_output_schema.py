from __future__ import annotations

from src.semantic_router.router.output_schema import (
    PredictionSourceMeta,
    build_prediction_output,
)
from src.semantic_router.router.schemas import RoutePrediction, ScoreItem


def test_build_prediction_output_contract():
    prediction = RoutePrediction(
        text="Исходный текст",
        normalized_text="Исходный текст",
        group_id="g1",
        group_name="Group 1",
        topic_id="g1.t1",
        topic_name="Topic 1",
        is_unknown=False,
        confidence=0.7,
        margin=0.2,
        decision_reason="matched",
        top_groups=[
            ScoreItem(
                id="g1",
                name="Group 1",
                score=0.6,
                group_id="g1",
                group_name="Group 1",
            )
        ],
        top_topics=[
            ScoreItem(
                id="g1.t1",
                name="Topic 1",
                score=0.7,
                group_id="g1",
                group_name="Group 1",
            )
        ],
    )

    output = build_prediction_output(
        route_prediction=prediction,
        source=PredictionSourceMeta(row_index=1, key="TXSUP-1"),
        chunks_count=1,
        encoder_model="hashing",
        calibration_applied=True,
        routing_config={"topic_min_similarity": 0.3},
    ).to_dict()

    assert output["schema_version"] == "semantic_router_prediction_v1"
    assert output["source"]["key"] == "TXSUP-1"
    assert output["prediction"]["topic_id"] == "g1.t1"
    assert output["scores"]["top_topics"][0]["score"] == 0.7
    assert output["debug"]["calibration_applied"] is True