from pathlib import Path

import cattrs
import polars as pl
import pyarrow.parquet as pq
from evals.types import InferenceOutput

from lalamo.evals.benchmark.callbacks import BaseBenchmarkCallbacks
from lalamo.evals.datasets.specs import REPO_TO_EVAL


def _validate_predictions_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Predictions file not found: {path}")
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

    df = df.rename({"model_output": "response"})
    predictions = cattrs.structure(df.to_dicts(), list[InferenceOutput])

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
    eval_spec = REPO_TO_EVAL[eval_name]
    eval_adapter = eval_spec.handler_type()

    callbacks.started(eval_spec.name, predictions_path)

    callbacks.loading_predictions()
    _validate_predictions_file(predictions_path)
    model_name, inference_engine = _extract_metadata(predictions_path)
    predictions = _load_predictions(predictions_path)

    callbacks.running_benchmark()
    metrics = eval_adapter.run_benchmark(
        predictions=predictions,
        eval_name=eval_spec.name,
        model_name=model_name,
        inference_engine=inference_engine,
    )

    callbacks.completed(metrics)
