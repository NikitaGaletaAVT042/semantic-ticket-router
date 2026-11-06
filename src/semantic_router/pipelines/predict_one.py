from __future__ import annotations

from src.semantic_router.router.semantic_router import SemanticTaxonomyRouter


def predict_one(
    *,
    router: SemanticTaxonomyRouter,
    text: str,
) -> dict:
    return router.predict(text).to_dict()