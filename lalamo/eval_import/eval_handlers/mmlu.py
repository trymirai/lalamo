import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from lalamo.eval_import.internal_format import InternalEvalRecord
from lalamo.eval_import.prediction_format import BenchmarkMetrics, PredictionRecord

from .common import EvalHandler
from .vendored.mmlu_pro import compute_accuracy_from_dir


@dataclass(frozen=True)
class MMLUProHandler(EvalHandler):
    def convert_record(self, record: dict) -> InternalEvalRecord:
        return InternalEvalRecord(
            id=str(record["question_id"]),
            question=record["question"],
            answer=record["answer"],
            options=record["options"],
            answer_index=record["answer_index"],
            reasoning=record.get("cot_content"),
            category=record.get("category"),
            metadata={
                "src": record.get("src", ""),
            },
        )

    def convert_split(self, parquet_path: Path) -> list[InternalEvalRecord]:
        table = pq.read_table(parquet_path)
        records = table.to_pydict()

        if not records:
            return []

        num_rows = len(next(iter(records.values())))
        return [self.convert_record({key: records[key][i] for key in records}) for i in range(num_rows)]

    def prepare_for_benchmark(
        self,
        predictions: list[PredictionRecord],
        ground_truth: list[InternalEvalRecord],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        by_category: dict[str, list[dict]] = {}
        for pred, gt in zip(predictions, ground_truth, strict=True):
            if pred.id != gt.id:
                raise ValueError(
                    f"ID mismatch at position: prediction.id={pred.id!r} != ground_truth.id={gt.id!r}. "
                    "Predictions and ground truth must be in the same order with matching IDs."
                )

            category = gt.category or "other"
            if category not in by_category:
                by_category[category] = []

            by_category[category].append({
                "model_outputs": pred.model_output,
                "answer": gt.answer,
            })

        for category, entries in by_category.items():
            category_file = output_dir / f"{category}.json"
            with open(category_file, "w") as f:
                json.dump(entries, f, indent=2)

        return output_dir

    def run_official_benchmark(
        self,
        prepared_data_path: Path,
        eval_name: str,
        model_name: str,
        split: str,
    ) -> BenchmarkMetrics:
        category_results = compute_accuracy_from_dir(prepared_data_path, level="l2")

        total_correct = sum(r["correct"] for r in category_results.values())
        total_examples = sum(r["total"] for r in category_results.values())
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        category_metrics = {category: r["accuracy"] for category, r in category_results.items()}

        return BenchmarkMetrics(
            eval_name=eval_name,
            model_name=model_name,
            split=split,
            overall_accuracy=overall_accuracy,
            total_examples=total_examples,
            correct=total_correct,
            incorrect=total_examples - total_correct,
            category_metrics=category_metrics,
        )
