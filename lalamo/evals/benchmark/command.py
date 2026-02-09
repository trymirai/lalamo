from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from evals.types import InferenceOutput

from lalamo.evals.benchmark.callbacks import BaseBenchmarkCallbacks
from lalamo.evals.datasets.specs import REPO_TO_EVAL


def _validate_predictions_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Predictions path is not a file: {path}")


def _load_predictions(path: Path) -> list[InferenceOutput]:
    df = pl.read_parquet(path)

    if df.is_empty():
        raise ValueError(f"Predictions file is empty: {path}")

    required_cols = {"id", "question", "model_output", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    predictions = [
        InferenceOutput(
            id=row["id"],
            response=row["model_output"],
            chain_of_thought=row.get("chain_of_thought"),
            question=row["question"],
            answer=row["answer"],
            metadata=row.get("metadata"),
        )
        for row in df.iter_rows(named=True)
    ]

    return predictions


def _extract_metadata(path: Path) -> tuple[str, str]:
    table = pq.read_table(path)
    schema_metadata = table.schema.metadata or {}

    if b"model_name" not in schema_metadata:
        raise ValueError("Predictions file missing required metadata: model_name")
    if b"inference_engine" not in schema_metadata:
        raise ValueError("Predictions file missing required metadata: inference_engine")

    model_name = schema_metadata[b"model_name"].decode()
    inference_engine = schema_metadata[b"inference_engine"].decode()

    return model_name, inference_engine


def benchmark_command_handler(
    eval_name: str,
    predictions_path: Path,
    callbacks: BaseBenchmarkCallbacks,
) -> None:
    if eval_name not in REPO_TO_EVAL:
        available = ", ".join(REPO_TO_EVAL.keys())
        raise ValueError(f"Unknown eval: {eval_name}. Available evals: {available}")

    eval_spec = REPO_TO_EVAL[eval_name]
    eval_adapter = eval_spec.handler_type()
    benchmark_split = eval_adapter.get_benchmark_split()

    callbacks.started(eval_spec.name, benchmark_split, predictions_path)

    callbacks.loading_predictions()
    _validate_predictions_file(predictions_path)
    model_name, inference_engine = _extract_metadata(predictions_path)
    predictions = _load_predictions(predictions_path)

    callbacks.preparing_benchmark()
    output_dir = predictions_path.parent / "benchmark_data"
    prepared_path = eval_adapter.prepare_for_benchmark(
        predictions=predictions,
        output_dir=output_dir,
    )

    callbacks.running_benchmark()
    metrics = eval_adapter.run_benchmark(
        prepared_data_path=prepared_path,
        eval_name=eval_spec.name,
        model_name=model_name,
        split=benchmark_split,
        inference_engine=inference_engine,
    )

    callbacks.completed(metrics)
