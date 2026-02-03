import importlib.metadata
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
from evals import BenchmarkMetrics, DatasetMetadata, InternalEvalRecord

from lalamo.evals.datasets.specs import EvalSpec

LALAMO_VERSION = importlib.metadata.version("lalamo")


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

    def converting_split(self, split: str) -> None:
        pass

    def finished_converting_split(self, split: str) -> None:
        pass

    def saving_dataset(self) -> None:
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
            filename.startswith((f"{split}/", f"{split}-"))
            or f"/{split}-" in filename
        )
    ]
    return parquet_files


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


def import_eval(
    eval_spec: EvalSpec,
    output_dir: Path,
    callbacks: EvalConversionCallbacks,
    max_examples: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = eval_spec.handler_type()
    total_examples: dict[str, int] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for split in eval_spec.splits:
            parquet_files = _list_parquet_files_for_split(eval_spec.repo, split)
            all_split_records: list[InternalEvalRecord] = []

            for filename in parquet_files:
                callbacks.downloading_file(filename)
                downloaded_path = huggingface_hub.hf_hub_download(
                    repo_id=eval_spec.repo,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(temp_path),
                )
                callbacks.finished_downloading_file(filename)

                callbacks.converting_split(split)
                internal_records = handler.convert_split(Path(downloaded_path))
                all_split_records.extend(internal_records)
                callbacks.finished_converting_split(split)

                # Stop early if we've reached max_examples
                if max_examples is not None and len(all_split_records) >= max_examples:
                    all_split_records = all_split_records[:max_examples]
                    break

            # Ensure we don't exceed max_examples after loop
            if max_examples is not None and len(all_split_records) > max_examples:
                all_split_records = all_split_records[:max_examples]

            callbacks.saving_dataset()
            total_examples[split] = len(all_split_records)
            internal_table = _records_to_table(all_split_records)
            output_parquet = output_dir / f"{split}.parquet"
            pq.write_table(internal_table, output_parquet)

    metadata = DatasetMetadata(
        lalamo_version=LALAMO_VERSION,
        name=eval_spec.name,
        repo=eval_spec.repo,
        splits=tuple(eval_spec.splits),
        schema_version="1.0",
        total_examples=total_examples,
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "lalamo_version": metadata.lalamo_version,
                "name": metadata.name,
                "repo": metadata.repo,
                "splits": list(metadata.splits),
                "schema_version": metadata.schema_version,
                "total_examples": metadata.total_examples,
            },
            f,
            indent=4,
        )


def benchmark_predictions(
    eval_spec: EvalSpec,
    model_name: str,
    predictions_file: Path,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
) -> "BenchmarkMetrics":
    """Orchestrate benchmarking workflow: prepare data + run official evaluation.

    Workflow:
        1. Load ground truth from internal format dataset
        2. Load predictions from parquet file
        3. Prepare data using handler.prepare_for_benchmark()
        4. Run benchmark using handler.run_official_benchmark(eval_spec.name, ...)
        5. Save metrics and annotated predictions
    """
    # handler = eval_spec.handler_type()
    # ground_truth = load_internal_dataset(dataset_dir / f"{split}.parquet")
    # predictions = load_predictions(predictions_file)
    # prepared_path = handler.prepare_for_benchmark(predictions, ground_truth, output_dir / "prepared")
    # metrics = handler.run_official_benchmark(prepared_path, eval_spec.name, model_name, split)
    # save_metrics(metrics, output_dir / "metrics.json")
    # return metrics
    raise NotImplementedError("benchmark_predictions orchestration not yet implemented")


def run_eval(
    eval_spec: EvalSpec,
    model_path: Path,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
) -> "BenchmarkMetrics":
    """Complete end-to-end evaluation workflow: inference + benchmarking.

    Workflow:
        1. Load internal format dataset
        2. Load model
        3. Run inference to generate predictions
        4. Run benchmarking on predictions
    """
    raise NotImplementedError("run_eval orchestration not yet implemented")


@dataclass
class InferenceCallbacks:
    model_path: Path
    dataset_path: Path
    output_path: Path

    def started(self) -> None:
        pass

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self) -> None:
        pass

    def running_inference(self, current: int, total: int) -> None:
        pass

    def finished_inference(self) -> None:
        pass

    def saving_predictions(self) -> None:
        pass

    def finished(self) -> None:
        pass


def run_inference(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    _max_output_length: int,
    _batch_size: int,
    callbacks: type[InferenceCallbacks],
) -> None:
    """Run model inference on eval dataset and save predictions.

    Workflow:
        1. Load model
        2. Load internal format dataset
        3. Run inference to generate predictions
        4. Save predictions to parquet
    """
    cb = callbacks(
        model_path=model_path,
        dataset_path=dataset_path,
        output_path=output_path,
    )
    cb.started()
    raise NotImplementedError("run_inference not yet implemented")


def convert_dataset(
    eval_spec: EvalSpec,
    output_dir: Path,
    callbacks_type: Callable[[EvalSpec, Path], EvalConversionCallbacks] = EvalConversionCallbacks,
    max_examples: int | None = None,
) -> None:
    callbacks = callbacks_type(eval_spec, output_dir)

    if output_dir.exists():
        callbacks.output_dir_exists()

    callbacks.started()

    import_eval(eval_spec, output_dir, callbacks, max_examples=max_examples)

    callbacks.finished()
