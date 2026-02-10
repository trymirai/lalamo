from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
from evals.types import InferenceEngineType, InternalEvalRecord

from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference.callbacks import BaseRunInferenceCallbacks
from lalamo.evals.inference.engines import (
    CustomAPIEngineConfig,
    CustomAPIInferenceEngine,
    LalamoEngineConfig,
    LalamoInferenceEngine,
)
from lalamo.evals.inference.formats import build_inference_dataframe, parse_inference_outputs


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
    dataset_dir: Path,
    output_dir: Path,
    engine_config: LalamoEngineConfig | CustomAPIEngineConfig,
    inference_overrides: dict[str, Any],
    limit: int | None = None,
    callbacks: BaseRunInferenceCallbacks | None = None,
) -> Path:
    if callbacks is None:
        callbacks = BaseRunInferenceCallbacks()

    eval_spec = REPO_TO_EVAL[eval_repo]
    eval_adapter = eval_spec.handler_type()

    if isinstance(engine_config, LalamoEngineConfig):
        engine_type = InferenceEngineType.LOCAL
        inference_engine = LalamoInferenceEngine
    elif isinstance(engine_config, CustomAPIEngineConfig):
        engine_type = InferenceEngineType.CUSTOM_API
        inference_engine = CustomAPIInferenceEngine
    else:
        raise TypeError(f"Unsupported engine config type: {type(engine_config)}")

    adapter_config = eval_adapter.get_inference_config(engine_type)
    overrides = {k: v for k, v in inference_overrides.items() if v is not None}
    inference_config = replace(adapter_config, **overrides)

    callbacks.started()
    callbacks.inference_config_loaded(asdict(adapter_config), overrides)

    inference_engine = inference_engine(engine_config, inference_config)

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
    input_df = build_inference_dataframe(
        prompts,
        benchmark_records,
        conversation_column=inference_engine.get_conversation_column_name(),
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_df.write_parquet(input_path)

    raw_output_path = output_dir / "inference_output.parquet"
    inference_engine.run_inference(input_path, raw_output_path, callbacks)

    output_df = pl.read_parquet(raw_output_path)
    input_df = pl.read_parquet(input_path)
    outputs = parse_inference_outputs(output_df, input_df)

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

    predictions_path = output_dir / "predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    table = predictions_df.to_arrow()
    file_metadata = {
        b"model_name": engine_config.get_model_name().encode(),
        b"inference_engine": engine_type.value.encode(),
        b"eval_name": eval_repo.encode(),
    }
    table = table.replace_schema_metadata(file_metadata)
    pq.write_table(table, predictions_path)

    callbacks.completed(predictions_path, len(outputs))

    return predictions_path
