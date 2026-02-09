from dataclasses import asdict, replace
from pathlib import Path

import polars as pl
from evals.types import InferenceConfig, InternalEvalRecord

from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference import LalamoInferenceEngine
from lalamo.evals.inference.callbacks import BaseRunInferenceCallbacks


def _load_internal_dataset(
    dataset_dir: Path,
    split: str,
    limit: int | None = None,
) -> list[InternalEvalRecord]:
    split_file = dataset_dir / f"{split}.parquet"
    df = pl.read_parquet(split_file)

    if limit:
        df = df.head(limit)

    records = [InternalEvalRecord(**row) for row in df.iter_rows(named=True)]
    return records


def infer_command_handler(
    eval_repo: str,
    model_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    inference_overrides: InferenceConfig,
    limit: int | None = None,
    batch_size: int | None = None,
    max_vram_bytes: int | None = None,
    engine: str = "lalamo",
    callbacks: BaseRunInferenceCallbacks | None = None,
) -> Path:
    if callbacks is None:
        callbacks = BaseRunInferenceCallbacks()

    eval_spec = REPO_TO_EVAL[eval_repo]
    eval_adapter = eval_spec.handler_type()

    adapter_config = eval_adapter.get_inference_config()
    overrides = {k: v for k, v in asdict(inference_overrides).items() if v is not None}
    inference_config = replace(adapter_config, **overrides)

    callbacks.started()
    callbacks.inference_config_loaded(asdict(adapter_config), overrides)

    if engine == "lalamo":
        inference_engine = LalamoInferenceEngine(
            model_path=model_path,
            inference_config=inference_config,
            max_vram=max_vram_bytes,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}. Supported: lalamo")

    output_dir.mkdir(parents=True, exist_ok=True)

    loading_config = eval_adapter.get_loading_config(limit)

    datasets = {
        config.split: _load_internal_dataset(dataset_dir, config.split, config.limit)
        for config in loading_config
    }

    prompts = eval_adapter.format_prompts(datasets)

    benchmark_split = eval_adapter.get_benchmark_split()
    benchmark_records = datasets[benchmark_split]

    input_path = output_dir / "inference_input.parquet"
    inference_engine.prepare_input(prompts, benchmark_records, input_path)

    raw_output_path = output_dir / "inference_output.parquet"
    inference_engine.run_inference(input_path, raw_output_path, callbacks)

    outputs = inference_engine.parse_output(raw_output_path, input_path)

    predictions_path = output_dir / "predictions.parquet"
    predictions_df = pl.DataFrame(
        {
            "id": [o.id for o in outputs],
            "question": [o.question for o in outputs],
            "model_output": [o.response for o in outputs],
            "chain_of_thought": [o.chain_of_thought for o in outputs],
            "answer": [o.answer for o in outputs],
            "metadata": [o.metadata for o in outputs],
        },
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.write_parquet(predictions_path)

    callbacks.completed(predictions_path, len(outputs))

    return predictions_path
