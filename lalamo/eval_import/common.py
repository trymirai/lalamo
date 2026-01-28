from dataclasses import dataclass
from pathlib import Path

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


def import_eval(
    eval_spec: EvalSpec,
    output_dir: Path,
    callbacks: EvalConversionCallbacks,
) -> None:
    # TODO: Implement download logic using huggingface_hub.hf_hub_download()
    # For each split in eval_spec.splits:
    #   - Download parquet files
    #   - Save to output_dir/
    pass
