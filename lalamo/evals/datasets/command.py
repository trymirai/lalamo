import importlib.metadata
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from evals import DatasetMetadata, EvalAdapter, InternalEvalRecord

from lalamo.evals.datasets.callbacks import BaseConversionCallbacks
from lalamo.evals.datasets.specs import REPO_TO_EVAL, EvalSpec

LALAMO_VERSION = importlib.metadata.version("lalamo")
DATASET_SCHEMA_VERSION = "1.0"


def _records_to_table(records: list[InternalEvalRecord]) -> pa.Table:
    data = {
        "id": [r.id for r in records],
        "question": [r.question for r in records],
        "answer": [r.answer for r in records],
        "options": [r.options for r in records],
        "answer_index": [r.answer_index for r in records],
        "reasoning": [r.reasoning for r in records],
        "category": [r.category for r in records],
        "metadata": [json.dumps(r.metadata) if r.metadata else None for r in records],
    }
    return pa.table(data)


def _download_and_convert_split(
    repo_id: str,
    split: str,
    handler: type[EvalAdapter],
    temp_dir: Path,
    callbacks: BaseConversionCallbacks,
) -> list[InternalEvalRecord]:
    callbacks.downloading_file(f"{split} split")
    records = handler.download_split(
        repo_id=repo_id,
        split=split,
        temp_dir=temp_dir,
    )
    callbacks.finished_downloading_file(f"{split} split")
    return records


def convert_dataset_handler(
    eval_repo: str,
    output_dir: Path,
    callbacks: BaseConversionCallbacks,
) -> None:
    eval_spec = REPO_TO_EVAL[eval_repo]
    if output_dir.exists():
        callbacks.output_dir_exists()

    callbacks.started()

    output_dir.mkdir(parents=True, exist_ok=True)

    handler = eval_spec.handler_type()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for split in eval_spec.splits:
            all_split_records = _download_and_convert_split(
                repo_id=eval_spec.repo,
                split=split,
                handler=handler,
                temp_dir=temp_path,
                callbacks=callbacks,
            )

            callbacks.saving_dataset()
            internal_table = _records_to_table(all_split_records)
            output_parquet = output_dir / f"{split}.parquet"
            pq.write_table(internal_table, output_parquet)

    metadata = DatasetMetadata(
        lalamo_version=LALAMO_VERSION,
        name=eval_spec.name,
        repo=eval_spec.repo,
        splits=eval_spec.splits,
        schema_version=DATASET_SCHEMA_VERSION,
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(metadata), f, indent=4)

    callbacks.finished()

