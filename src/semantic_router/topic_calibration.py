from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_YAML_CALIBRATION_PATH = PROJECT_ROOT / "configs" / "topic_calibration.yaml"
DEFAULT_JSON_CALIBRATION_PATH = PROJECT_ROOT / "configs" / "topic_calibration.json"


def load_topic_calibration(path: str | Path | None = None) -> dict[str, float]:
    """
    Loads additive topic score calibration.

    Production format:

    topics:
      topic_id:
        bias: -0.03
        reason: "false positive magnet"

    Legacy JSON flat format is also supported:

    {
      "Topic name": -0.05
    }

    Returned dict keys may be topic_id or legacy topic_name.
    semantic_router should first check topic_id, then fallback to topic_name.
    """
    calibration_path = _resolve_calibration_path(path)

    if calibration_path is None or not calibration_path.exists():
        return {}

    raw = _read_calibration_file(calibration_path)

    if not isinstance(raw, dict):
        raise ValueError(f"Topic calibration must be object: {calibration_path}")

    if "topics" in raw:
        return _parse_structured_calibration(raw, calibration_path)

    return _parse_flat_calibration(raw, calibration_path)


def apply_topic_calibration(
    scores: dict[str, float],
    calibration: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Pure helper for tests and small scripts.
    """
    if calibration is None:
        calibration = load_topic_calibration()

    adjusted_scores = dict(scores)

    for topic_key, bias in calibration.items():
        if topic_key in adjusted_scores:
            adjusted_scores[topic_key] = float(adjusted_scores[topic_key]) + float(bias)

    return adjusted_scores


def _resolve_calibration_path(path: str | Path | None) -> Path | None:
    if path is not None:
        calibration_path = Path(path)

        if not calibration_path.is_absolute():
            calibration_path = PROJECT_ROOT / calibration_path

        return calibration_path

    if DEFAULT_YAML_CALIBRATION_PATH.exists():
        return DEFAULT_YAML_CALIBRATION_PATH

    if DEFAULT_JSON_CALIBRATION_PATH.exists():
        return DEFAULT_JSON_CALIBRATION_PATH

    return None


def _read_calibration_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(text) or {}
    else:
        value = json.loads(text)

    if not isinstance(value, dict):
        raise ValueError(f"Calibration file must contain object: {path}")

    return value


def _parse_structured_calibration(
    raw: dict[str, Any],
    path: Path,
) -> dict[str, float]:
    topics = raw.get("topics")

    if not isinstance(topics, dict):
        raise ValueError(f"Expected 'topics' object in calibration: {path}")

    result: dict[str, float] = {}

    for topic_id, payload in topics.items():
        normalized_topic_id = str(topic_id).strip()

        if not normalized_topic_id:
            continue

        if isinstance(payload, dict):
            raw_bias = payload.get("bias", 0.0)
        else:
            raw_bias = payload

        try:
            result[normalized_topic_id] = float(raw_bias)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Bad calibration bias for topic {normalized_topic_id!r}: {raw_bias!r}"
            ) from exc

    return result


def _parse_flat_calibration(
    raw: dict[str, Any],
    path: Path,
) -> dict[str, float]:
    result: dict[str, float] = {}

    for key, value in raw.items():
        topic_key = str(key).strip()

        if not topic_key:
            continue

        try:
            result[topic_key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Bad calibration value for topic {topic_key!r}: {value!r} in {path}"
            ) from exc

    return result