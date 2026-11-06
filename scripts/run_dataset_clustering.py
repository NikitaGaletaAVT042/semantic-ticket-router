from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_router.pipelines.predict_dataset_clusters import predict_dataset_clusters
from src.semantic_router.router.config import load_router_config
from src.semantic_router.router.semantic_router import SemanticTaxonomyRouter


CONFIG_PATH = PROJECT_ROOT / "configs" / "router_config.yaml"
INPUT_PATH = PROJECT_ROOT / "datasets_s" / "processed" / "marked_txsup_records.json"

CLUSTERS_OUTPUT_PATH = PROJECT_ROOT / "output" / "taxonomy_clusters.json"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "output" / "taxonomy_predictions.jsonl"

INCLUDE_UNKNOWN = True


def main() -> None:
    print("=" * 80)
    print("Semantic taxonomy dataset clustering")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config:       {CONFIG_PATH}")
    print(f"Input:        {INPUT_PATH}")
    print(f"Clusters out: {CLUSTERS_OUTPUT_PATH}")
    print(f"Predictions:  {PREDICTIONS_OUTPUT_PATH}")
    print()

    config = load_router_config(CONFIG_PATH)

    taxonomy_path = Path(config.taxonomy_path)

    if not taxonomy_path.is_absolute():
        taxonomy_path = PROJECT_ROOT / taxonomy_path

    # RouterConfig frozen=True, поэтому не мутируем объект,
    # а создаем новую копию с исправленным taxonomy_path.
    config = replace(config, taxonomy_path=str(taxonomy_path))

    print("Loading router...")
    router = SemanticTaxonomyRouter.from_config(config)

    print("Running dataset prediction and aggregation...")
    clusters = predict_dataset_clusters(
        router=router,
        input_path=INPUT_PATH,
        clusters_output_path=CLUSTERS_OUTPUT_PATH,
        predictions_output_path=PREDICTIONS_OUTPUT_PATH,
        include_unknown=INCLUDE_UNKNOWN,
    )

    total_items = sum(cluster["examples_count"] for cluster in clusters)

    print()
    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Total routed items: {total_items}")
    print(f"Total clusters:      {len(clusters)}")
    print()
    print("Top clusters:")

    for cluster in clusters[:20]:
        print(
            f"- {cluster['human_cluster_name']}: "
            f"{cluster['examples_count']} "
            f"(confidence={cluster['label_confidence']})"
        )

    print()
    print(f"Saved clusters to:    {CLUSTERS_OUTPUT_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()