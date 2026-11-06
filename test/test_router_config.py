from __future__ import annotations

import yaml

from src.semantic_router.router.config import load_router_config


def test_load_router_config_with_extended_routing_fields(tmp_path):
    config_path = tmp_path / "router_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "taxonomy_path": "taxonomy/taxonomy_v1.json",
                "embedding": {
                    "use_hashing_encoder": True,
                    "hashing_dim": 128,
                },
                "routing": {
                    "group_min_similarity": 0.2,
                    "group_min_margin": 0.03,
                    "topic_min_similarity": 0.35,
                    "topic_min_margin": 0.04,
                    "topic_high_confidence": 0.5,
                    "topic_high_confidence_margin": 0.08,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    config = load_router_config(config_path)

    assert config.taxonomy_path == "taxonomy/taxonomy_v1.json"
    assert config.embedding.use_hashing_encoder is True
    assert config.embedding.hashing_dim == 128
    assert config.routing.group_min_similarity == 0.2
    assert config.routing.group_min_margin == 0.03
    assert config.routing.topic_min_similarity == 0.35
    assert config.routing.topic_min_margin == 0.04
    assert config.routing.topic_high_confidence == 0.5
    assert config.routing.topic_high_confidence_margin == 0.08