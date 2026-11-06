from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")

    return value


def short_text(value: Any, limit: int = 600) -> str:
    if not isinstance(value, str):
        return ""

    text = value.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze evaluation errors from markup_eval_report.json"
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("output/eval_exact_no_process.json"),
        help="Path to evaluation report JSON",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many top rows to print",
    )

    parser.add_argument(
        "--low-margin-threshold",
        type=float,
        default=0.03,
        help="Margin threshold for unstable wrong predictions",
    )

    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.65,
        help="Confidence threshold for confident wrong predictions",
    )

    args = parser.parse_args()

    report_path = resolve_project_path(args.report_path)
    report = read_json(report_path)

    errors = report.get("errors", [])
    if not isinstance(errors, list):
        raise RuntimeError(f"Expected report['errors'] to be list in {report_path}")

    by_expected: Counter[str] = Counter()
    by_predicted: Counter[str] = Counter()
    by_pair: Counter[tuple[str, str]] = Counter()

    low_margin_wrong: list[dict[str, Any]] = []
    high_confidence_wrong: list[dict[str, Any]] = []

    for error in errors:
        if not isinstance(error, dict):
            continue

        expected = str(error.get("expected") or "")
        predicted = str(error.get("predicted") or "")

        by_expected[expected] += 1
        by_predicted[predicted] += 1
        by_pair[(expected, predicted)] += 1

        confidence = error.get("confidence")
        margin = error.get("margin")

        if isinstance(margin, (int, float)) and margin < args.low_margin_threshold:
            low_margin_wrong.append(error)

        if isinstance(confidence, (int, float)) and confidence >= args.high_confidence_threshold:
            high_confidence_wrong.append(error)

    print()
    print("REPORT")
    print(f"path:     {report_path.relative_to(PROJECT_ROOT)}")
    print(f"total:    {report.get('total_compared')}")
    print(f"correct:  {report.get('correct')}")
    print(f"accuracy: {report.get('accuracy')}")

    if "accepted_total" in report:
        print()
        print("REJECT METRICS")
        print(f"accepted_total:    {report.get('accepted_total')}")
        print(f"accepted_correct:  {report.get('accepted_correct')}")
        print(f"accepted_accuracy: {report.get('accepted_accuracy')}")
        print(f"rejected_total:    {report.get('rejected_total')}")
        print(f"coverage:          {report.get('coverage')}")

    print()
    print("=== WORST EXPECTED LABELS ===")
    for label, count in by_expected.most_common(args.top):
        print(f"{count:>4}  {label}")

    print()
    print("=== PREDICTION MAGNETS ===")
    for label, count in by_predicted.most_common(args.top):
        print(f"{count:>4}  {label}")

    print()
    print("=== TOP CONFUSION PAIRS ===")
    for (expected, predicted), count in by_pair.most_common(args.top):
        print(f"{count:>4}  {expected} -> {predicted}")

    print()
    print("=== LOW MARGIN WRONG CASES ===")
    print(f"count: {len(low_margin_wrong)}")
    for error in low_margin_wrong[:15]:
        print()
        print(f"{error.get('key')}: {error.get('expected')} -> {error.get('predicted')}")
        print(f"confidence={error.get('confidence')} margin={error.get('margin')}")
        print(short_text(error.get("text")))

    print()
    print("=== HIGH CONFIDENCE WRONG CASES ===")
    print(f"count: {len(high_confidence_wrong)}")
    for error in high_confidence_wrong[:15]:
        print()
        print(f"{error.get('key')}: {error.get('expected')} -> {error.get('predicted')}")
        print(f"confidence={error.get('confidence')} margin={error.get('margin')}")
        print(short_text(error.get("text")))


if __name__ == "__main__":
    main()