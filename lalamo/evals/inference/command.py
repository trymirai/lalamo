from pathlib import Path

import polars as pl
from evals.types import InferenceOutput, InternalEvalRecord

from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference import LalamoInferenceEngine
from lalamo.evals.inference.callbacks import BaseRunInferenceCallbacks


def _load_internal_dataset(
    dataset_dir: Path,
    split: str,
    max_examples: int | None,
    category: str | None = None,
) -> list[InternalEvalRecord]:
    split_file = dataset_dir / f"{split}.parquet"
    df = pl.read_parquet(split_file)

    if category:
        df = df.filter(pl.col("category") == category)

    if max_examples:
        df = df.head(max_examples)

    records = [InternalEvalRecord(**row) for row in df.iter_rows(named=True)]
    return records


def infer_command_handler(
    eval_repo: str,
    model_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    callbacks: BaseRunInferenceCallbacks,
    engine: str = "lalamo",
    num_few_shot: int = 5,
    max_examples: int | None = None,
    category: str | None = None,
    batch_size: int | None = None,
    vram_gb: float | None = None,
    max_output_length: int = 2048,
) -> Path:
    eval_spec = REPO_TO_EVAL[eval_repo]
    if engine == "lalamo":
        inference_engine = LalamoInferenceEngine(
            model_path=model_path,
            max_vram=int(vram_gb * 1024**3) if vram_gb else None,
            batch_size=batch_size,
            max_output_length=max_output_length,
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}. Supported: lalamo")

    eval_adapter = eval_spec.handler_type()

    callbacks.started()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get splits from adapter
    inference_split = eval_adapter.get_inference_split()
    few_shot_split = eval_adapter.get_few_shot_split()

    callbacks.loading_test_dataset()
    test_records = _load_internal_dataset(dataset_dir, inference_split, max_examples, category)

    few_shot_records = None
    if num_few_shot > 0 and few_shot_split is not None:
        callbacks.loading_validation_dataset()
        few_shot_records = _load_internal_dataset(dataset_dir, few_shot_split, None)
    else:
        callbacks.skipped_validation_dataset()

    callbacks.formatting_prompts()
    prompts = eval_adapter.format_prompts(
        records=test_records,
        few_shot_source=few_shot_records,
        num_few_shot=num_few_shot,
    )

    callbacks.preparing_input()
    input_path = output_dir / "inference_input.parquet"
    inference_engine.prepare_input(prompts, input_path)

    callbacks.running_inference()
    raw_output_path = output_dir / "inference_output.parquet"
    inference_engine.run_inference(input_path, raw_output_path, callbacks)

    callbacks.parsing_output()
    outputs = inference_engine.parse_output(raw_output_path, input_path)

    predictions_path = output_dir / f"predictions_{inference_split}.parquet"
    predictions_df = pl.DataFrame(
        {
            "id": [o.id for o in outputs],
            "model_output": [o.response for o in outputs],
            "chain_of_thought": [o.chain_of_thought for o in outputs],
        },
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.write_parquet(predictions_path)

    callbacks.completed(predictions_path, len(outputs))

    return predictions_path
