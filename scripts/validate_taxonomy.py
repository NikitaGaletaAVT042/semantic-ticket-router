from __future__ import annotations

import argparse
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_router.taxonomy.loader import load_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", required=True)
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)

    groups_count = len(taxonomy.groups)
    topics_count = sum(len(group.topics) for group in taxonomy.groups)

    print("Taxonomy is valid")
    print(f"Version: {taxonomy.taxonomy_version}")
    print(f"Groups: {groups_count}")
    print(f"Topics: {topics_count}")


if __name__ == "__main__":
    main()