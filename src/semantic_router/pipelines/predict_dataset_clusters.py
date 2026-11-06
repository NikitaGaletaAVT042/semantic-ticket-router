from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.semantic_router.router.semantic_router import SemanticTaxonomyRouter


CLIENT_TEXT_FIELDS = (
    "client_messages_hashed",
    "client_messages",
)

ORIGINAL_DIALOG_FIELDS = (
    "dialog_hashed",
    "dialog_original",
)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """
    Загружает JSON-файл с обращениями.

    Ожидаемый формат:
    [
      {
        "data": {
          "key": "TXSUP-...",
          "client_messages": [...],
          ...
        }
      }
    ]

    Также поддерживаем вариант, если data-обертки нет.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"Dataset must be a JSON list, got: {type(payload).__name__}")

    rows: list[dict[str, Any]] = []

    for item in payload:
        if not isinstance(item, dict):
            continue

        data = item.get("data", item)

        if isinstance(data, dict):
            rows.append(data)

    return rows


def extract_txsup_key(row: dict[str, Any]) -> str:
    key = row.get("key")
    if isinstance(key, str) and key.strip():
        return key.strip()

    issue_key = row.get("issue_key")
    if isinstance(issue_key, str) and issue_key.strip():
        return issue_key.strip()

    return ""


def extract_client_messages(row: dict[str, Any]) -> list[str]:
    """
    Берем только сообщения клиента.

    Почему не берем operator_messages:
    оператор часто уже содержит ответ/подсказку,
    а нам нужно классифицировать именно проблему пользователя.
    """
    for field_name in CLIENT_TEXT_FIELDS:
        value = row.get(field_name)

        if isinstance(value, list):
            messages = [str(message).strip() for message in value if str(message).strip()]
            if messages:
                return messages

    return []


def extract_dialog_user_messages(row: dict[str, Any]) -> list[str]:
    """
    Fallback, если client_messages нет.
    Достаем user-сообщения из dialog_original/dialog_hashed.
    """
    for field_name in ORIGINAL_DIALOG_FIELDS:
        dialog = row.get(field_name)

        if not isinstance(dialog, list):
            continue

        messages: list[str] = []

        for message in dialog:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip().lower()
            text = str(message.get("text", "")).strip()

            if role == "user" and text:
                messages.append(text)

        if messages:
            return messages

    return []


def build_text_for_routing(row: dict[str, Any]) -> str:
    """
    Собирает текст обращения для роутера.

    Берем все клиентские сообщения, потому что в TXSUP часто проблема
    раскрывается не в первом сообщении, а в продолжении диалога.
    """
    messages = extract_client_messages(row)

    if not messages:
        messages = extract_dialog_user_messages(row)

    return "\n".join(messages).strip()


def make_short_example(text: str, max_len: int = 180) -> str:
    text = " ".join(str(text).split())

    if len(text) <= max_len:
        return text

    return text[: max_len - 3].rstrip() + "..."


def prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """
    Совместимость на случай, если RoutePrediction:
    - dataclass
    - имеет метод to_dict()
    - обычный объект
    """
    if hasattr(prediction, "to_dict"):
        result = prediction.to_dict()
        if isinstance(result, dict):
            return result

    if is_dataclass(prediction):
        return asdict(prediction)

    if hasattr(prediction, "__dict__"):
        return dict(prediction.__dict__)

    raise TypeError(f"Unsupported prediction type: {type(prediction).__name__}")


def get_prediction_field(prediction: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(field_name, default)

    return getattr(prediction, field_name, default)


def build_prediction_rows(
    router: SemanticTaxonomyRouter,
    dataset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Прогоняет каждое обращение через router.

    Возвращает плоский список:
    [
      {
        "key": "TXSUP-...",
        "text": "...",
        "prediction": {...}
      }
    ]
    """
    result_rows: list[dict[str, Any]] = []

    for index, row in enumerate(dataset_rows, start=1):
        key = extract_txsup_key(row)
        text = build_text_for_routing(row)

        if not text:
            continue

        prediction = router.predict(text)
        prediction_dict = prediction_to_dict(prediction)

        result_rows.append(
            {
                "row_index": index,
                "key": key,
                "text": text,
                "prediction": prediction_dict,
            }
        )

    return result_rows


def aggregate_predictions_to_clusters(
    prediction_rows: list[dict[str, Any]],
    *,
    include_unknown: bool = True,
    cluster_id_prefix: str = "business",
) -> list[dict[str, Any]]:
    """
    Собирает предсказания в cluster JSON.

    Группировка идет по topic_id.
    Если topic_id unknown и include_unknown=False, такие строки пропускаются.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in prediction_rows:
        prediction = row["prediction"]

        topic_id = prediction.get("topic_id", "")
        is_unknown = bool(prediction.get("is_unknown", False))

        if is_unknown and not include_unknown:
            continue

        if not topic_id:
            continue

        buckets[topic_id].append(row)

    clusters: list[dict[str, Any]] = []

    for topic_id, rows in buckets.items():
        first_prediction = rows[0]["prediction"]

        topic_name = first_prediction.get("topic_name") or topic_id
        group_id = first_prediction.get("group_id")
        group_name = first_prediction.get("group_name")

        examples = [row["text"] for row in rows if row.get("text")]
        txsup_keys = [row["key"] for row in rows if row.get("key")]

        confidences = [
            float(row["prediction"].get("confidence", 0.0))
            for row in rows
            if row.get("prediction") is not None
        ]

        margins = [
            float(row["prediction"].get("margin", 0.0))
            for row in rows
            if row.get("prediction") is not None
        ]

        original_0 = examples[0] if len(examples) > 0 else ""
        original_1 = examples[1] if len(examples) > 1 else ""
        original_2 = examples[2] if len(examples) > 2 else ""

        examples_count = len(rows)
        avg_confidence = round(mean(confidences), 4) if confidences else 0.0
        avg_margin = round(mean(margins), 4) if margins else 0.0

        cluster = {
            "cluster_id": f"{cluster_id_prefix}::{topic_name}",
            "coarse_domain": "business",
            "scenario_family": "business",
            "fine_scenario": None,
            "human_cluster_name": topic_name,
            "examples_count": examples_count,
            "example": make_short_example(original_0),
            "original_client_text": original_0,
            "original_client_text_1": original_1,
            "original_client_text_2": original_2,
            "txsup_keys": txsup_keys,
            "feature_stats": {},
            "route_source": "semantic_taxonomy_router",
            "label_confidence": avg_confidence,
            "second_pass_source_cluster": None,
            "cluster_name": topic_name,
            "is_mixed_cluster": False,
            "top_level_intent": "business",
            "taxonomy_meta": {
                "group_id": group_id,
                "group_name": group_name,
                "topic_id": topic_id,
                "topic_name": topic_name,
                "avg_confidence": avg_confidence,
                "avg_margin": avg_margin,
                "is_unknown_cluster": bool(first_prediction.get("is_unknown", False)),
                "decision_reason": first_prediction.get("decision_reason"),
            },
        }

        clusters.append(cluster)

    clusters.sort(key=lambda item: item["examples_count"], reverse=True)

    return clusters


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def predict_dataset_clusters(
    *,
    router: SemanticTaxonomyRouter,
    input_path: str | Path,
    clusters_output_path: str | Path,
    predictions_output_path: str | Path | None = None,
    include_unknown: bool = True,
) -> list[dict[str, Any]]:
    dataset_rows = load_dataset(input_path)

    prediction_rows = build_prediction_rows(
        router=router,
        dataset_rows=dataset_rows,
    )

    clusters = aggregate_predictions_to_clusters(
        prediction_rows,
        include_unknown=include_unknown,
    )

    save_json(clusters_output_path, clusters)

    if predictions_output_path is not None:
        save_jsonl(predictions_output_path, prediction_rows)

    return clusters