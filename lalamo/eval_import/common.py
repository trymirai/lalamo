from dataclasses import dataclass
from pathlib import Path

import huggingface_hub

from .eval_specs.common import EvalSpec


@dataclass
class EvalConversionCallbacks:
    eval_spec: EvalSpec
    output_dir: Path

    def output_dir_exists(self) -> None:
        pass

    def started(self) -> None:
        pass

    def downloading_file(self, filename: str) -> None:
        pass

    def finished_downloading_file(self, filename: str) -> None:
        pass

    def finished(self) -> None:
        pass


def _list_parquet_files_for_split(repo_id: str, split: str) -> list[str]:
    all_files = huggingface_hub.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [
        filename
        for filename in all_files
        if filename.endswith(".parquet")
        and (
            filename.startswith(f"{split}/")
            or filename.startswith(f"{split}-")
            or f"/{split}-" in filename
        )
    ]
    return parquet_files


def import_eval(
    eval_spec: EvalSpec,
    output_dir: Path,
    callbacks: EvalConversionCallbacks,
) -> None:
    """
    Download eval dataset from HuggingFace.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in eval_spec.splits:
        parquet_files = _list_parquet_files_for_split(eval_spec.repo, split)

        for filename in parquet_files:
            callbacks.downloading_file(filename)
            huggingface_hub.hf_hub_download(
                repo_id=eval_spec.repo,
                repo_type="dataset",
                filename=filename,
                local_dir=str(output_dir),
            )
            callbacks.finished_downloading_file(filename)
