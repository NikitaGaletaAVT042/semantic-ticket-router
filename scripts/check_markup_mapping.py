from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_against_markup import (
    DEFAULT_MARKUP_PATH,
    DEFAULT_TAXONOMY_PATH,
    build_topic_name_to_ids,
    normalize_label,
    read_markup,
    resolve_expected_topic_id,
)
from src.semantic_router.taxonomy.loader import load_taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markup", type=Path, default=DEFAULT_MARKUP_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    topic_name_to_ids = build_topic_name_to_ids(taxonomy)
    expected_by_key = read_markup(args.markup)

    label_counts: Counter[str] = Counter(expected_by_key.values())

    unmapped: list[tuple[str, int, str]] = []

    for label, count in label_counts.most_common():
        topic_id = resolve_expected_topic_id(
            label=label,
            topic_name_to_ids=topic_name_to_ids,
        )

        if topic_id.startswith("__UNMAPPED_LABEL__::"):
            unmapped.append((label, count, topic_id))

    print()
    print("=" * 80)
    print("MARKUP MAPPING CHECK")
    print("=" * 80)
    print(f"Total labels:    {len(label_counts)}")
    print(f"Unmapped labels: {len(unmapped)}")

    if unmapped:
        print()
        print("UNMAPPED")
        for label, count, topic_id in unmapped:
            print(f"{count:>4}  {label} -> {topic_id}")
    else:
        print()
        print("All markup labels are mapped to taxonomy topic_id.")


if __name__ == "__main__":
    main()