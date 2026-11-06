from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    use_hashing_encoder: bool = False
    hashing_dim: int = 384
    batch_size: int = 32


@dataclass(frozen=True)
class RoutingConfig:
    active_statuses: list[str] = field(default_factory=lambda: ["active"])

    group_min_similarity: float = 0.18
    group_min_margin: float = 0.015

    topic_min_similarity: float = 0.32
    topic_min_margin: float = 0.02
    reject_on_low_margin: bool = True

    topic_high_confidence: float = 0.48
    topic_high_confidence_margin: float = 0.05

    negative_penalty: float = 0.08
    topic_score_top_k: int = 2
    group_prior_weight: float = 0.08

    use_chunk_scoring: bool = True
    chunk_score_weight: float = 0.35
    max_chunks: int = 8
    min_chunk_chars: int = 40
    max_chunk_chars: int = 700

    return_top_k_groups: int = 3
    return_top_k_topics: int = 5


@dataclass(frozen=True)
class RouterConfig:
    taxonomy_path: str
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


def load_router_config(path: str | Path) -> RouterConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return RouterConfig(
        taxonomy_path=str(raw.get("taxonomy_path", "taxonomy/taxonomy_v1.json")),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        routing=RoutingConfig(**raw.get("routing", {})),
    )