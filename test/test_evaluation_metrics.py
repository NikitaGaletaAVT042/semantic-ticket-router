from __future__ import annotations

from src.semantic_router.evaluation.metrics import EvalItem, evaluate_predictions


def test_evaluate_predictions_basic_metrics():
    report = evaluate_predictions(
        [
            EvalItem(
                key="1",
                expected_topic_id="g1.t1",
                predicted_topic_id="g1.t1",
                expected_group_id="g1",
                predicted_group_id="g1",
                is_unknown=False,
                top_topic_ids=["g1.t1", "g1.t2"],
            ),
            EvalItem(
                key="2",
                expected_topic_id="g1.t2",
                predicted_topic_id="g1.t1",
                expected_group_id="g1",
                predicted_group_id="g1",
                is_unknown=False,
                top_topic_ids=["g1.t1", "g1.t2"],
            ),
            EvalItem(
                key="3",
                expected_topic_id="g2.t1",
                predicted_topic_id="unknown",
                expected_group_id="g2",
                predicted_group_id="unknown",
                is_unknown=True,
                top_topic_ids=["g1.t1"],
            ),
        ]
    )

    assert report.total == 3
    assert report.topic_accuracy == 0.3333
    assert report.group_accuracy == 0.6667
    assert report.accepted_total == 2
    assert report.coverage == 0.6667
    assert report.accepted_accuracy == 0.5
    assert report.top_3_topic_accuracy == 0.6667
    assert report.confusion_matrix["g1.t2"]["g1.t1"] == 1


def test_evaluate_predictions_empty_input():
    report = evaluate_predictions([])

    assert report.total == 0
    assert report.topic_accuracy == 0.0
    assert report.macro_f1 == 0.0
    assert report.confusion_matrix == {}