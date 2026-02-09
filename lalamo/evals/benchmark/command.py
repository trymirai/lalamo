from pathlib import Path

import polars as pl
from evals.types import InferenceOutput

from lalamo.evals.benchmark.callbacks import BaseBenchmarkCallbacks
from lalamo.evals.datasets.specs import REPO_TO_EVAL


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
    pred_df = pl.read_parquet(predictions_path)

    predictions = [
        InferenceOutput(
            id=row["id"],
            response=row["model_output"],
            chain_of_thought=row.get("chain_of_thought"),
            question=row["question"],
            answer=row["answer"],
            metadata=row.get("metadata"),
        )
        for row in pred_df.iter_rows(named=True)
    ]

    # TODO(mullakhmetov): reconsider how to extract model_name from predictions
    model_name = pred_df["model_name"][0] if "model_name" in pred_df.columns else "unknown"

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
    )

    callbacks.completed(metrics)
