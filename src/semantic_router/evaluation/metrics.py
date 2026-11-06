from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class EvalItem:
    key: str
    expected_topic_id: str
    predicted_topic_id: str
    expected_group_id: str
    predicted_group_id: str
    is_unknown: bool
    top_topic_ids: list[str] = field(default_factory=list)

    confidence: float | None = None
    margin: float | None = None
    decision_reason: str | None = None
    top_topics: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class LabelMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    overall_accuracy: float
    group_accuracy: float
    topic_accuracy: float
    accepted_total: int
    accepted_accuracy: float
    coverage: float
    macro_f1: float
    weighted_f1: float
    unknown_precision: float
    unknown_recall: float
    unknown_f1: float
    top_3_topic_accuracy: float
    top_5_topic_accuracy: float
    per_label: list[LabelMetrics]
    confusion_matrix: dict[str, dict[str, int]]


def evaluate_predictions(items: Iterable[EvalItem]) -> EvaluationReport:
    rows = list(items)
    total = len(rows)

    if total == 0:
        return EvaluationReport(
            total=0,
            overall_accuracy=0.0,
            group_accuracy=0.0,
            topic_accuracy=0.0,
            accepted_total=0,
            accepted_accuracy=0.0,
            coverage=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            unknown_precision=0.0,
            unknown_recall=0.0,
            unknown_f1=0.0,
            top_3_topic_accuracy=0.0,
            top_5_topic_accuracy=0.0,
            per_label=[],
            confusion_matrix={},
        )

    topic_correct = sum(
        1 for row in rows
        if row.expected_topic_id == row.predicted_topic_id
    )
    group_correct = sum(
        1 for row in rows
        if row.expected_group_id == row.predicted_group_id
    )

    accepted_rows = [row for row in rows if not row.is_unknown]
    accepted_correct = sum(
        1 for row in accepted_rows
        if row.expected_topic_id == row.predicted_topic_id
    )

    top_3_correct = sum(
        1 for row in rows
        if row.expected_topic_id in row.top_topic_ids[:3]
    )
    top_5_correct = sum(
        1 for row in rows
        if row.expected_topic_id in row.top_topic_ids[:5]
    )

    labels = sorted(
        {
            row.expected_topic_id
            for row in rows
        }
        | {
            row.predicted_topic_id
            for row in rows
        }
    )

    per_label = _compute_per_label_metrics(rows, labels)
    macro_f1 = _safe_mean([item.f1 for item in per_label])

    support_sum = sum(item.support for item in per_label)
    weighted_f1 = (
        sum(item.f1 * item.support for item in per_label) / support_sum
        if support_sum
        else 0.0
    )

    unknown_precision, unknown_recall, unknown_f1 = _compute_unknown_metrics(rows)
    confusion_matrix = _build_confusion_matrix(rows)

    return EvaluationReport(
        total=total,
        overall_accuracy=_round(topic_correct / total),
        group_accuracy=_round(group_correct / total),
        topic_accuracy=_round(topic_correct / total),
        accepted_total=len(accepted_rows),
        accepted_accuracy=_round(accepted_correct / len(accepted_rows)) if accepted_rows else 0.0,
        coverage=_round(len(accepted_rows) / total),
        macro_f1=_round(macro_f1),
        weighted_f1=_round(weighted_f1),
        unknown_precision=unknown_precision,
        unknown_recall=unknown_recall,
        unknown_f1=unknown_f1,
        top_3_topic_accuracy=_round(top_3_correct / total),
        top_5_topic_accuracy=_round(top_5_correct / total),
        per_label=per_label,
        confusion_matrix=confusion_matrix,
    )


def _compute_per_label_metrics(
    rows: list[EvalItem],
    labels: list[str],
) -> list[LabelMetrics]:
    result: list[LabelMetrics] = []

    for label in labels:
        tp = sum(
            1 for row in rows
            if row.expected_topic_id == label and row.predicted_topic_id == label
        )
        fp = sum(
            1 for row in rows
            if row.expected_topic_id != label and row.predicted_topic_id == label
        )
        fn = sum(
            1 for row in rows
            if row.expected_topic_id == label and row.predicted_topic_id != label
        )
        support = sum(1 for row in rows if row.expected_topic_id == label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        result.append(
            LabelMetrics(
                label=label,
                precision=_round(precision),
                recall=_round(recall),
                f1=_round(f1),
                support=support,
            )
        )

    return result


def _compute_unknown_metrics(rows: list[EvalItem]) -> tuple[float, float, float]:
    expected_unknown = {
        "unknown",
    }

    tp = sum(
        1 for row in rows
        if row.predicted_topic_id in expected_unknown
        and row.expected_topic_id in expected_unknown
    )
    fp = sum(
        1 for row in rows
        if row.predicted_topic_id in expected_unknown
        and row.expected_topic_id not in expected_unknown
    )
    fn = sum(
        1 for row in rows
        if row.predicted_topic_id not in expected_unknown
        and row.expected_topic_id in expected_unknown
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return _round(precision), _round(recall), _round(f1)


def _build_confusion_matrix(rows: list[EvalItem]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        matrix[row.expected_topic_id][row.predicted_topic_id] += 1

    return {
        expected: dict(predicted_counts)
        for expected, predicted_counts in matrix.items()
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _round(value: float) -> float:
    return round(float(value), 4)