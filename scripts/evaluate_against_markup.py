from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.semantic_router.evaluation.metrics import EvalItem, evaluate_predictions
from src.semantic_router.taxonomy.loader import load_taxonomy
from src.semantic_router.taxonomy.schema import Taxonomy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MARKUP_PATH = PROJECT_ROOT / "data" / "markup" / "txsup_markup.txt"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "output" / "taxonomy_predictions.jsonl"
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "taxonomy" / "taxonomy_v1.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "markup_eval_report.json"

TXSUP_RE = re.compile(r"TXSUP-\d+", re.IGNORECASE)

PROCESS_LABELS = {
    "Экспертная проверка",
    "Экспертная проверка пакета",
    "Ответ предоставлен в телефонном обращении",
}

ALIASES = {
    "Экспертная проверка": "Экспертная проверка пакета",
    "Уточнение фрагментом Смежного ЗУ": "Уточнение фрагментов Смежного ЗУ",
    "Уточнение фрагментов Смежного ЗУ": "Уточнение фрагментов Смежного ЗУ",
    "Уточнение границы/зоны": "Границы/зоны",
    "Системы координат и кадастровые кварталы": "Системы координат",
    "Импорт и экспорт файлов с координатами": "Импорт и экспорт файлов с координатами",
    "Экспорт файлов с координатами": "Импорт и экспорт файлов с координатами",
    "Настройки программы, интерфейс": "Настройки программы, интерфейс",
    "Доступность функций/регламентов - новое название для \"Вкладки, разделы и доступность функций\"": "Доступность функций/регламентов",
}
LABEL_TO_TOPIC_ID_OVERRIDES = {
    "Ошибки импорта пакета": "import_export_exchange.packages",
}

def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("ё", "е").replace("Ё", "Е")
    return ALIASES.get(text, text)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_markup(path: Path) -> dict[str, str]:
    key_to_label: dict[str, str] = {}
    current_label = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        keys = TXSUP_RE.findall(line)

        if keys:
            if not current_label:
                continue

            for key in keys:
                key_to_label[key.upper()] = normalize_label(current_label)

            continue

        current_label = normalize_label(line)

    return key_to_label


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Bad JSONL line {line_no} in {path}: {exc}") from exc

        if isinstance(value, dict):
            rows.append(value)

    return rows


def build_topic_name_to_ids(taxonomy: Taxonomy) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    for group in taxonomy.groups:
        result.setdefault(normalize_label(group.unknown_topic.topic_name), []).append(
            group.unknown_topic.topic_id
        )

        for topic in group.topics:
            result.setdefault(normalize_label(topic.topic_name), []).append(topic.topic_id)

    return result


def build_topic_id_to_group_id(taxonomy: Taxonomy) -> dict[str, str]:
    result: dict[str, str] = {}

    for group in taxonomy.groups:
        result[group.unknown_topic.topic_id] = group.group_id

        for topic in group.topics:
            result[topic.topic_id] = group.group_id

    return result


def resolve_expected_topic_id(
    *,
    label: str,
    topic_name_to_ids: dict[str, list[str]],
) -> str:
    normalized = normalize_label(label)

    override_topic_id = LABEL_TO_TOPIC_ID_OVERRIDES.get(normalized)
    if override_topic_id is not None:
        return override_topic_id

    topic_ids = topic_name_to_ids.get(normalized, [])

    if len(topic_ids) == 1:
        return topic_ids[0]

    if len(topic_ids) > 1:
        return topic_ids[0]

    return f"__UNMAPPED_LABEL__::{normalized}"


def extract_prediction(row: dict[str, Any]) -> dict[str, Any]:
    prediction = row.get("prediction")

    if isinstance(prediction, dict):
        return prediction

    return {}


def extract_key(row: dict[str, Any]) -> str:
    candidates = [
        row.get("key"),
        row.get("txsup_key"),
        row.get("issue_key"),
        row.get("id"),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        match = TXSUP_RE.search(str(candidate))
        if match:
            return match.group(0).upper()

    for value in row.values():
        if isinstance(value, str):
            match = TXSUP_RE.search(value)
            if match:
                return match.group(0).upper()

    return ""


def build_prediction_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for row in rows:
        key = extract_key(row)

        if not key:
            continue

        result[key] = row

    return result


def build_eval_items(
    *,
    expected_by_key: dict[str, str],
    predictions_by_key: dict[str, dict[str, Any]],
    taxonomy: Taxonomy,
) -> tuple[list[EvalItem], list[str]]:
    topic_name_to_ids = build_topic_name_to_ids(taxonomy)
    topic_id_to_group_id = build_topic_id_to_group_id(taxonomy)

    items: list[EvalItem] = []
    missing_predictions: list[str] = []

    for key, expected_label in expected_by_key.items():
        expected_topic_id = resolve_expected_topic_id(
            label=expected_label,
            topic_name_to_ids=topic_name_to_ids,
        )
        expected_group_id = topic_id_to_group_id.get(expected_topic_id, "unknown")

        prediction_row = predictions_by_key.get(key)

        if prediction_row is None:
            missing_predictions.append(key)
            items.append(
                EvalItem(
                    key=key,
                    expected_topic_id=expected_topic_id,
                    predicted_topic_id="unknown",
                    expected_group_id=expected_group_id,
                    predicted_group_id="unknown",
                    is_unknown=True,
                    top_topic_ids=[],
                )
            )
            continue

        prediction = extract_prediction(prediction_row)

        predicted_topic_id = str(prediction.get("topic_id") or "unknown")
        predicted_group_id = str(prediction.get("group_id") or "unknown")
        is_unknown = bool(prediction.get("is_unknown", False))
        confidence = prediction.get("confidence")
        margin = prediction.get("margin")
        decision_reason = prediction.get("decision_reason")

        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None

        try:
            margin_value = float(margin) if margin is not None else None
        except (TypeError, ValueError):
            margin_value = None
        top_topics_raw = prediction.get("top_topics") or []
        top_topic_ids: list[str] = []
        top_topics: list[dict[str, object]] = []

        if isinstance(top_topics_raw, list):
            for item in top_topics_raw:
                if not isinstance(item, dict):
                    continue

                topic_id = item.get("id")
                topic_name = item.get("name")
                topic_score = item.get("score")
                topic_group_id = item.get("group_id")
                topic_group_name = item.get("group_name")

                if topic_id:
                    top_topic_ids.append(str(topic_id))

                try:
                    score_value = float(topic_score) if topic_score is not None else None
                except (TypeError, ValueError):
                    score_value = None

                top_topics.append(
                    {
                        "id": str(topic_id or ""),
                        "name": str(topic_name or ""),
                        "score": score_value,
                        "group_id": str(topic_group_id or ""),
                        "group_name": str(topic_group_name or ""),
                    }
                )

        if isinstance(top_topics_raw, list):
            for item in top_topics_raw:
                if isinstance(item, dict):
                    topic_id = item.get("id")
                    if topic_id:
                        top_topic_ids.append(str(topic_id))

        items.append(
            EvalItem(
                key=key,
                expected_topic_id=expected_topic_id,
                predicted_topic_id=predicted_topic_id,
                expected_group_id=expected_group_id,
                predicted_group_id=predicted_group_id,
                is_unknown=is_unknown,
                top_topic_ids=top_topic_ids,
                confidence=confidence_value,
                margin=margin_value,
                decision_reason=str(decision_reason) if decision_reason is not None else None,
                top_topics=top_topics,
            )
        )

    return items, missing_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic router predictions against txsup markup."
    )

    parser.add_argument(
        "--markup",
        type=Path,
        default=DEFAULT_MARKUP_PATH,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--exclude-process-labels",
        action="store_true",
        help="Exclude expert check / phone answer labels from evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    markup_path = resolve_path(args.markup)
    predictions_path = resolve_path(args.predictions)
    taxonomy_path = resolve_path(args.taxonomy)
    report_path = resolve_path(args.report_path)

    taxonomy = load_taxonomy(taxonomy_path)

    expected_by_key = read_markup(markup_path)

    if args.exclude_process_labels:
        expected_by_key = {
            key: label
            for key, label in expected_by_key.items()
            if normalize_label(label) not in PROCESS_LABELS
        }

    prediction_rows = read_jsonl(predictions_path)
    predictions_by_key = build_prediction_by_key(prediction_rows)

    eval_items, missing_predictions = build_eval_items(
        expected_by_key=expected_by_key,
        predictions_by_key=predictions_by_key,
        taxonomy=taxonomy,
    )

    report = evaluate_predictions(eval_items)

    payload = {
        "mode": {
            "exclude_process_labels": args.exclude_process_labels,
        },
        "paths": {
            "markup": str(markup_path.relative_to(PROJECT_ROOT)),
            "predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
            "taxonomy": str(taxonomy_path.relative_to(PROJECT_ROOT)),
        },
        "metrics": asdict(report),
        "missing_predictions_count": len(missing_predictions),
        "missing_predictions": missing_predictions,
        "errors": [
            asdict(item)
            for item in eval_items
            if item.expected_topic_id != item.predicted_topic_id
        ][:300],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    print(f"Total:              {report.total}")
    print(f"Topic accuracy:     {report.topic_accuracy}")
    print(f"Group accuracy:     {report.group_accuracy}")
    print(f"Accepted total:     {report.accepted_total}")
    print(f"Accepted accuracy:  {report.accepted_accuracy}")
    print(f"Coverage:           {report.coverage}")
    print(f"Macro F1:           {report.macro_f1}")
    print(f"Weighted F1:        {report.weighted_f1}")
    print(f"Unknown precision:  {report.unknown_precision}")
    print(f"Unknown recall:     {report.unknown_recall}")
    print(f"Unknown F1:         {report.unknown_f1}")
    print(f"Top-3 accuracy:     {report.top_3_topic_accuracy}")
    print(f"Top-5 accuracy:     {report.top_5_topic_accuracy}")
    print(f"Missing preds:      {len(missing_predictions)}")
    print()
    print(f"Saved report:       {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()