from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "eval_exact_no_process_back_to_global.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "errors_by_predicted_label.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-examples-per-label",
        type=int,
        default=10,
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def short_text(text: str, limit: int = 900) -> str:
    text = " ".join(str(text).split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def main() -> None:
    args = parse_args()

    report_path = resolve_path(args.report_path)
    output_path = resolve_path(args.output_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[dict[str, Any]] = report.get("errors", [])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for error in errors:
        predicted = str(
            error.get("predicted_label")
            or error.get("predicted")
            or ""
        )

        if not predicted or predicted == "__REJECTED__":
            continue

        grouped[predicted].append(error)

    lines: list[str] = []
    lines.append("# Errors by predicted label")
    lines.append("")
    lines.append(f"Report: `{report_path.relative_to(PROJECT_ROOT)}`")
    lines.append("")

    sorted_items = sorted(
        grouped.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for predicted_label, items in sorted_items:
        if len(items) < args.min_count:
            continue

        lines.append(f"## {predicted_label}")
        lines.append("")
        lines.append(f"Ошибочных предсказаний: **{len(items)}**")
        lines.append("")

        expected_counts: dict[str, int] = defaultdict(int)
        for item in items:
            expected = str(
                item.get("expected_label")
                or item.get("expected")
                or ""
            )
            expected_counts[expected] += 1

        lines.append("Ожидаемые классы, которые сюда ошибочно попали:")
        lines.append("")

        for expected, count in sorted(expected_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- `{expected}`: {count}")

        lines.append("")

        for index, item in enumerate(items[: args.max_examples_per_label], start=1):
            key = item.get("key", "")
            expected = item.get("expected_label") or item.get("expected")
            confidence = item.get("confidence")
            margin = item.get("margin")
            text = short_text(str(item.get("text") or ""))

            lines.append(f"### {index}. {key}")
            lines.append("")
            lines.append(f"- expected: `{expected}`")
            lines.append(f"- predicted: `{predicted_label}`")
            lines.append(f"- confidence: `{confidence}`")
            lines.append(f"- margin: `{margin}`")
            lines.append("")
            lines.append(text)
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()