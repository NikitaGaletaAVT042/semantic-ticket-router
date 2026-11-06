from __future__ import annotations

from src.semantic_router.io.jsonl import read_jsonl, write_jsonl
from src.semantic_router.router.semantic_router import SemanticTaxonomyRouter


def predict_batch(
    *,
    router: SemanticTaxonomyRouter,
    input_path: str,
    output_path: str,
    text_field: str = "text",
) -> None:
    rows = read_jsonl(input_path)

    output_rows: list[dict] = []
    for row in rows:
        text = str(row.get(text_field, ""))
        prediction = router.predict(text).to_dict()

        output_rows.append(
            {
                **row,
                "prediction": prediction,
            }
        )

    write_jsonl(output_path, output_rows)