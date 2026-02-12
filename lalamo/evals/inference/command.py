from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cattrs
import polars as pl
import pyarrow.parquet as pq

from lalamo.evals.datasets.specs import EVAL_ADAPTERS
from lalamo.evals.inference.callbacks import BaseRunInferenceCallbacks
from lalamo.evals.inference.engines import (
    CustomAPIEngineConfig,
    CustomAPIInferenceEngine,
    LalamoEngineConfig,
    LalamoInferenceEngine,
)
from lalamo.evals.inference.formats import build_inference_dataframe, parse_inference_outputs


def infer_command_handler(
    eval_name: str,
    output_dir: Path,
    engine_config: LalamoEngineConfig | CustomAPIEngineConfig,
    inference_overrides: dict[str, Any],
    limit: int | None = None,
    callbacks: BaseRunInferenceCallbacks | None = None,
) -> Path:
    if callbacks is None:
        callbacks = BaseRunInferenceCallbacks()

    adapter_class = EVAL_ADAPTERS[eval_name]
    eval_adapter = adapter_class()

    if isinstance(engine_config, LalamoEngineConfig):
        inference_engine_cls = LalamoInferenceEngine
    elif isinstance(engine_config, CustomAPIEngineConfig):
        inference_engine_cls = CustomAPIInferenceEngine
    else:
        raise TypeError(f"Unsupported engine config type: {type(engine_config)}")

    adapter_config = eval_adapter.get_inference_config()
    overrides = {k: v for k, v in inference_overrides.items() if v is not None}
    inference_config = replace(adapter_config, **overrides)

    callbacks.started()
    callbacks.inference_config_loaded(asdict(adapter_config), overrides)

    inference_engine = inference_engine_cls(engine_config, inference_config)

    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = eval_adapter.format_prompts(limit=limit)

    input_path = output_dir / "inference_input.parquet"
    input_df = build_inference_dataframe(
        prompts,
        conversation_column=inference_engine.get_conversation_column_name(),
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_df.write_parquet(input_path)

    raw_output_path = output_dir / "inference_output.parquet"
    inference_engine.run_inference(input_path, raw_output_path, callbacks)

    output_df = pl.read_parquet(raw_output_path)
    input_df = pl.read_parquet(input_path)
    outputs = parse_inference_outputs(output_df, input_df)

    predictions_df = pl.DataFrame([cattrs.unstructure(o) for o in outputs])
    if len(outputs) > 0:
        predictions_df = predictions_df.rename({"response": "model_output"})

    predictions_path = output_dir / "predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    table = predictions_df.to_arrow()
    file_metadata = {
        b"model_name": engine_config.get_model_name().encode(),
        b"inference_engine": inference_engine.get_engine_name().encode(),
        b"eval_name": eval_name.encode(),
    }
    table = table.replace_schema_metadata(file_metadata)
    pq.write_table(table, predictions_path)

    callbacks.completed(predictions_path, len(outputs))

    return predictions_path
