from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "eval_after_decision_policy.json",
    )
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report_path

    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    report = read_json(report_path)
    metrics = report.get("metrics", {})
    errors = report.get("errors", [])

    if not isinstance(errors, list):
        raise ValueError("Expected report['errors'] to be list")

    confusion: Counter[tuple[str, str]] = Counter()
    error_kind_counter: Counter[str] = Counter()
    expected_counter: Counter[str] = Counter()
    predicted_counter: Counter[str] = Counter()
    rank_counter: Counter[str] = Counter()

    examples_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for error in errors:
        if not isinstance(error, dict):
            continue

        expected_topic = str(error.get("expected_topic_id") or "")
        predicted_topic = str(error.get("predicted_topic_id") or "")
        expected_group = str(error.get("expected_group_id") or "")
        predicted_group = str(error.get("predicted_group_id") or "")
        top_topic_ids = error.get("top_topic_ids") or []

        confusion[(expected_topic, predicted_topic)] += 1
        expected_counter[expected_topic] += 1
        predicted_counter[predicted_topic] += 1

        if expected_group != predicted_group:
            error_kind = "wrong_group"
        elif predicted_topic == "unknown" or predicted_topic.endswith(".unknown"):
            error_kind = "group_unknown_instead_of_topic"
        else:
            error_kind = "correct_group_wrong_topic"

        error_kind_counter[error_kind] += 1

        if isinstance(top_topic_ids, list) and expected_topic in top_topic_ids:
            rank = top_topic_ids.index(expected_topic) + 1
            rank_counter[f"rank_{rank}"] += 1
        else:
            rank_counter["not_in_top"] += 1

        pair = (expected_topic, predicted_topic)
        if len(examples_by_pair[pair]) < 3:
            examples_by_pair[pair].append(error)

    print()
    print("=" * 80)
    print("DIAGNOSTIC REPORT")
    print("=" * 80)

    print()
    print("METRICS")
    for key in [
        "total",
        "topic_accuracy",
        "group_accuracy",
        "accepted_accuracy",
        "coverage",
        "macro_f1",
        "weighted_f1",
        "top_3_topic_accuracy",
        "top_5_topic_accuracy",
    ]:
        print(f"{key}: {metrics.get(key)}")

    print()
    print("ERROR KINDS")
    for kind, count in error_kind_counter.most_common():
        print(f"{count:>4}  {kind}")

    print()
    print("EXPECTED TOPICS WITH MOST ERRORS")
    for label, count in expected_counter.most_common(args.top):
        print(f"{count:>4}  {label}")

    print()
    print("PREDICTED TOPICS AS ERROR MAGNETS")
    for label, count in predicted_counter.most_common(args.top):
        print(f"{count:>4}  {label}")

    print()
    print("EXPECTED TOPIC RANK IN TOP-K FOR ERRORS")
    for rank, count in rank_counter.most_common():
        print(f"{count:>4}  {rank}")

    print()
    print("TOP CONFUSION PAIRS")
    for (expected, predicted), count in confusion.most_common(args.top):
        print()
        print(f"{count:>4}  {expected} -> {predicted}")

        for example in examples_by_pair[(expected, predicted)]:
            key = example.get("key", "")
            expected_group = example.get("expected_group_id", "")
            predicted_group = example.get("predicted_group_id", "")
            top_topic_ids = example.get("top_topic_ids", [])
            top_topics = example.get("top_topics", [])

            print(f"      top_topics: {top_topic_ids[:5]}")

            if isinstance(top_topics, list) and top_topics:
                print("      top_scores:")
                for topic in top_topics[:5]:
                    if not isinstance(topic, dict):
                        continue

                    topic_id = topic.get("id")
                    score = topic.get("score")
                    group_id = topic.get("group_id")

                    print(f"        - {topic_id} score={score} group={group_id}")

            print(f"      key={key}")
            print(f"      groups: {expected_group} -> {predicted_group}")
            print(f"      top_topics: {top_topic_ids[:5]}")

    print()
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()