from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.semantic_router.embeddings.encoder import SentenceTransformerEncoder
from src.semantic_router.supervised.dataset_loader import iter_dataset_records
from src.semantic_router.supervised.labeled_index import LabeledExample, LabeledKnnIndex
from src.semantic_router.supervised.markup_loader import load_key_to_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markup", default="data/markup/txsup_markup.txt")
    parser.add_argument("--input-dir", default="datasets_s/processed")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--output", default="output/labeled_knn_eval.json")
    args = parser.parse_args()

    key_to_label = load_key_to_label(Path(args.markup))

    records = list(iter_dataset_records(Path(args.input_dir)))
    records_by_key = {item["key"]: item for item in records}

    examples: list[LabeledExample] = []
    missing_records: list[str] = []

    for key, label in key_to_label.items():
        record = records_by_key.get(key)

        if record is None:
            missing_records.append(key)
            continue

        examples.append(
            LabeledExample(
                key=key,
                label=label,
                text=record["text"],
            )
        )

    encoder = SentenceTransformerEncoder(model_name=args.model)
    index = LabeledKnnIndex.build(examples=examples, encoder=encoder)

    correct = 0
    total = 0

    errors = []
    confusion: Counter[tuple[str, str]] = Counter()

    for i, example in enumerate(examples):
        query_vector = index.vectors[i]

        neighbors = index.search_with_vector(
            query_vector=query_vector,
            top_k=args.top_k + 1,
        )

        # leave-one-out: убираем самого себя
        neighbors = [n for n in neighbors if n.key != example.key][: args.top_k]

        label_scores: Counter[str] = Counter()
        for neighbor in neighbors:
            label_scores[neighbor.label] += max(0.0, neighbor.score)

        if not label_scores:
            predicted = None
            confidence = 0.0
        else:
            predicted, score_sum = label_scores.most_common(1)[0]
            confidence = score_sum / sum(label_scores.values())

        total += 1

        if predicted == example.label:
            correct += 1
        else:
            confusion[(example.label, str(predicted))] += 1
            errors.append(
                {
                    "key": example.key,
                    "expected": example.label,
                    "predicted": predicted,
                    "confidence": round(confidence, 4),
                    "neighbors": [
                        {
                            "key": n.key,
                            "label": n.label,
                            "score": round(n.score, 4),
                        }
                        for n in neighbors
                    ],
                    "text": example.text,
                }
            )

    report = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "missing_records": missing_records,
        "top_confusions": [
            {
                "expected": expected,
                "predicted": predicted,
                "count": count,
            }
            for (expected, predicted), count in confusion.most_common(50)
        ],
        "errors": errors[:100],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Examples:  {len(examples)}")
    print(f"Accuracy:  {report['accuracy']}")
    print(f"Saved:     {output_path}")

    print()
    print("TOP CONFUSIONS:")
    for row in report["top_confusions"][:30]:
        print(f"- {row['expected']} -> {row['predicted']}: {row['count']}")


if __name__ == "__main__":
    main()