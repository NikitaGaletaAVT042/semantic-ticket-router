from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "datasets_s" / "processed"
MARKUP_PATH = PROJECT_ROOT / "data" / "markup" / "txsup_markup.txt"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "marked_txsup_records.json"
MISSING_PATH = OUTPUT_DIR / "missing_marked_txsup.json"

TXSUP_RE = re.compile(r"TXSUP-\d+", re.IGNORECASE)
BATCH_FILE_RE = re.compile(r"batch_0000[0-4]\.json$")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_marked_txsup_keys(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Markup file not found: {path}")

    text = path.read_text(encoding="utf-8")
    return {match.upper() for match in TXSUP_RE.findall(text)}


def find_batch_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and BATCH_FILE_RE.match(path.name)
    ]

    if not files:
        raise FileNotFoundError(
            f"No batch_00000.json ... batch_00004.json files found in {input_dir}"
        )

    return sorted(files)


def get_record_key(record: dict[str, Any]) -> str | None:
    data = record.get("data")

    if not isinstance(data, dict):
        return None

    key = data.get("key")

    if not isinstance(key, str):
        return None

    match = TXSUP_RE.search(key)

    if match is None:
        return None

    return match.group(0).upper()


def main() -> None:
    print("=" * 80)
    print("Collect marked TXSUP records")
    print("=" * 80)

    marked_keys = read_marked_txsup_keys(MARKUP_PATH)
    batch_files = find_batch_files(INPUT_DIR)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input dir:    {INPUT_DIR}")
    print(f"Markup:       {MARKUP_PATH}")
    print(f"Output:       {OUTPUT_PATH}")
    print()
    print(f"Marked TXSUP keys: {len(marked_keys)}")
    print()

    result_records: list[dict[str, Any]] = []
    found_keys: set[str] = set()
    duplicate_keys: set[str] = set()

    for batch_path in batch_files:
        batch_data = read_json(batch_path)

        if not isinstance(batch_data, list):
            raise ValueError(
                f"Expected JSON array in {batch_path}, got {type(batch_data).__name__}"
            )

        added_from_file = 0

        for record in batch_data:
            if not isinstance(record, dict):
                continue

            key = get_record_key(record)

            if key is None:
                continue

            if key not in marked_keys:
                continue

            if key in found_keys:
                duplicate_keys.add(key)
                continue

            # ВАЖНО:
            # добавляем исходную запись целиком, без изменений
            result_records.append(record)
            found_keys.add(key)
            added_from_file += 1

        print(f"{batch_path.name}: added {added_from_file}")

    missing_keys = sorted(marked_keys - found_keys)

    write_json(OUTPUT_PATH, result_records)
    write_json(MISSING_PATH, missing_keys)

    print()
    print("Done.")
    print(f"Saved original records: {len(result_records)}")
    print(f"Found keys:             {len(found_keys)}")
    print(f"Missing keys:           {len(missing_keys)}")
    print(f"Duplicate keys skipped: {len(duplicate_keys)}")
    print()
    print(f"Output file:            {OUTPUT_PATH}")
    print(f"Missing file:           {MISSING_PATH}")


if __name__ == "__main__":
    main()