from pathlib import Path

from lalamo.evals.datasets import REPO_TO_EVAL, convert_dataset
from lalamo.evals.datasets.callbacks import BaseConversionCallbacks


def convert_dataset_handler(
    eval_repo: str,
    output_dir: Path,
    callbacks: BaseConversionCallbacks,
) -> None:
    eval_spec = REPO_TO_EVAL[eval_repo]
    convert_dataset(eval_spec, output_dir, callbacks)
