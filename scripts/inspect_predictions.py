from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="output/taxonomy_predictions.jsonl",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        required=True,
    )
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    wanted = {key.upper() for key in args.keys}

    found = 0

    for row in rows:
        key = str(row.get("key", "")).upper()

        if key not in wanted:
            continue

        found += 1
        prediction = row.get("prediction", {})

        print("=" * 100)
        print(key)
        print("-" * 100)
        print("PREDICTED:")
        print(f"  group: {prediction.get('group_name')}")
        print(f"  topic: {prediction.get('topic_name')}")
        print(f"  confidence: {prediction.get('confidence')}")
        print(f"  margin: {prediction.get('margin')}")
        print(f"  reason: {prediction.get('decision_reason')}")
        print()
        print("TEXT:")
        print(str(row.get("text", ""))[:1200])
        print()
        print("TOP TOPICS:")

        for index, item in enumerate(prediction.get("top_topics", []), start=1):
            print(
                f"  {index}. {item.get('name')} "
                f"[score={item.get('score')}, group={item.get('group_name')}]"
            )

        print()
        print("TOP GROUPS:")

        for index, item in enumerate(prediction.get("top_groups", []), start=1):
            print(
                f"  {index}. {item.get('name')} "
                f"[score={item.get('score')}]"
            )

    missing = wanted - {
        str(row.get("key", "")).upper()
        for row in rows
    }

    if missing:
        print("=" * 100)
        print("MISSING KEYS:")
        for key in sorted(missing):
            print(f"- {key}")

    print("=" * 100)
    print(f"Found: {found}")


if __name__ == "__main__":
    main()