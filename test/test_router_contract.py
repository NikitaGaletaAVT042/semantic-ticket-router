from __future__ import annotations

import json

from src.semantic_router.router.config import EmbeddingConfig, RouterConfig, RoutingConfig
from src.semantic_router.router.semantic_router import SemanticTaxonomyRouter


def test_router_predict_contract(tmp_path):
    taxonomy_path = tmp_path / "taxonomy.json"

    taxonomy_path.write_text(
        json.dumps(
            {
                "taxonomy_version": "test",
                "language": "ru",
                "description": "test",
                "groups": [
                    {
                        "group_id": "documents",
                        "group_name": "Документы",
                        "group_description": "Документы и исходные данные",
                        "unknown_topic": {
                            "topic_id": "documents.unknown",
                            "topic_name": "UNKNOWN: Документы",
                            "is_unknown": True,
                        },
                        "topics": [
                            {
                                "topic_id": "documents.add_documents",
                                "topic_name": "Добавление документов",
                                "topic_description": "Добавление документов в исходные данные",
                                "positive_examples": [
                                    "Как добавить документ в исходные данные?",
                                    "Не могу прикрепить документ.",
                                ],
                                "negative_examples": [
                                    "Не строится окружность.",
                                    "Пакет не отправляется.",
                                ],
                                "status": "active",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config = RouterConfig(
        taxonomy_path=str(taxonomy_path),
        embedding=EmbeddingConfig(use_hashing_encoder=True, hashing_dim=128),
        routing=RoutingConfig(
            group_min_similarity=0.01,
            topic_min_similarity=0.01,
            topic_min_margin=0.0,
            active_statuses=["active"],
        ),
    )

    router = SemanticTaxonomyRouter.from_config(config)
    prediction = router.predict("Не могу добавить документ в исходные данные")

    assert prediction.group_id == "documents"
    assert prediction.topic_id in {"documents.add_documents", "documents.unknown"}
    assert isinstance(prediction.confidence, float)
    assert prediction.top_groups
    assert prediction.top_topics