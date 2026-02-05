from pathlib import Path

from lalamo.evals.benchmark.callbacks import BaseBenchmarkCallbacks
from lalamo.evals.benchmark.runner import compute_metrics
from lalamo.evals.datasets.specs import REPO_TO_EVAL


def benchmark_command_handler(
    eval_name: str,
    predictions_path: Path,
    dataset_dir: Path,
    callbacks: BaseBenchmarkCallbacks,
) -> None:
    if eval_name not in REPO_TO_EVAL:
        available = ", ".join(REPO_TO_EVAL.keys())
        raise ValueError(f"Unknown eval: {eval_name}. Available evals: {available}")

    compute_metrics(
        eval_spec=REPO_TO_EVAL[eval_name],
        predictions_path=predictions_path,
        dataset_dir=dataset_dir,
        callbacks=callbacks,
    )
