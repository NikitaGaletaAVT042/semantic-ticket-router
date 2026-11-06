from __future__ import annotations

import argparse

from src.semantic_router.pipelines.predict_batch import predict_batch
from src.semantic_router.router.config import load_router_config
from src.semantic_router.router.semantic_router import SemanticTaxonomyRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/router_config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    config = load_router_config(args.config)
    router = SemanticTaxonomyRouter.from_config(config)

    predict_batch(
        router=router,
        input_path=args.input,
        output_path=args.output,
        text_field=args.text_field,
    )

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()