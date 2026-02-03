from pathlib import Path

import polars as pl
from evals.types import BenchmarkMetrics, InternalEvalRecord, PredictionRecord

from lalamo.evals.benchmark.callbacks import BaseBenchmarkCallbacks
from lalamo.evals.datasets.specs import EvalSpec


def compute_metrics(
    eval_spec: EvalSpec,
    predictions_path: Path,
    dataset_dir: Path,
    split: str,
    callbacks: BaseBenchmarkCallbacks,
) -> tuple[Path, BenchmarkMetrics]:
    eval_adapter = eval_spec.handler_type()

    callbacks.started(eval_spec.name, split, predictions_path)

    callbacks.loading_predictions()
    pred_df = pl.read_parquet(predictions_path)
    predictions = [
        PredictionRecord(
            id=row["id"],
            model_output=row["model_output"],
        )
        for row in pred_df.iter_rows(named=True)
    ]

    # TODO(mullakhmetov): reconsider how to extract model_name from predictions
    model_name = pred_df["model_name"][0] if "model_name" in pred_df.columns else "unknown"

    callbacks.loading_ground_truth()
    gt_path = dataset_dir / f"{split}.parquet"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    gt_df = pl.read_parquet(gt_path)
    prediction_ids = {p.id for p in predictions}
    ground_truth = [InternalEvalRecord(**row) for row in gt_df.iter_rows(named=True) if row["id"] in prediction_ids]

    # TODO(mullakhmetov): think about avoiding sorting predictions and ground truth
    predictions.sort(key=lambda p: p.id)
    ground_truth.sort(key=lambda g: g.id)

    callbacks.preparing_benchmark()
    output_dir = predictions_path.parent / "benchmark_data"
    prepared_path = eval_adapter.prepare_for_benchmark(
        predictions=predictions,
        ground_truth=ground_truth,
        output_dir=output_dir,
    )

    callbacks.running_benchmark()
    metrics = eval_adapter.run_benchmark(
        prepared_data_path=prepared_path,
        eval_name=eval_spec.name,
        model_name=model_name,
        split=split,
    )

    callbacks.completed(metrics)

    return prepared_path, metrics
