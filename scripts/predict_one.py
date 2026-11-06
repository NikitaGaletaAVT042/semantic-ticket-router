from __future__ import annotations

import argparse
import json

from src.semantic_router.router.config import load_router_config
from src.semantic_router.router.semantic_router import SemanticTaxonomyRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/router_config.yaml")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    config = load_router_config(args.config)
    router = SemanticTaxonomyRouter.from_config(config)

    prediction = router.predict(args.text)
    print(json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()