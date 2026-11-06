from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def iter_dataset_records(input_dir: Path):
    for path in sorted(input_dir.glob("batch_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, dict):
            items = data.get("items") or data.get("data") or []
        else:
            items = data

        for item in items:
            record = item.get("data") if isinstance(item, dict) and "data" in item else item

            if not isinstance(record, dict):
                continue

            key = str(record.get("key") or "").upper()
            if not key:
                continue

            text = extract_client_text(record)
            if not text:
                continue

            yield {
                "key": key,
                "text": text,
                "record": record,
            }


def extract_client_text(record: dict[str, Any]) -> str:
    messages = record.get("client_messages_hashed") or record.get("client_messages") or []

    if isinstance(messages, list) and messages:
        return "\n\n".join(str(message) for message in messages if message)

    value = record.get("original_client_text") or record.get("text") or ""
    return str(value)