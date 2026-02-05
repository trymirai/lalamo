from pathlib import Path

import polars as pl
from evals.protocols import EvalAdapter
from evals.types import InferenceOutput, InternalEvalRecord

from lalamo.common import get_default_device_bytes
from lalamo.data.huggingface_message import HFMessage, load_hf_parquet
from lalamo.evals.inference.callbacks import BaseRunInferenceCallbacks
from lalamo.evals.inference.engines import InferenceEngine
from lalamo.message_processor import AssistantMessage
from lalamo.models.common import InferenceConfig
from lalamo.models.language_model import LanguageModelConfig


def run_inference(
    dataset_dir: Path,
    output_dir: Path,
    inference_engine: InferenceEngine,
    eval_adapter: EvalAdapter,
    num_few_shot: int,
    max_examples: int | None,
    category: str | None,
    callbacks: BaseRunInferenceCallbacks,
) -> Path:
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
    _save_predictions(outputs, predictions_path)
    callbacks.completed(predictions_path, len(outputs))

    return predictions_path


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


def _save_predictions(
    outputs: list[InferenceOutput],
    output_path: Path,
) -> None:
    df = pl.DataFrame(
        {
            "id": [o.id for o in outputs],
            "model_output": [o.response for o in outputs],
            "chain_of_thought": [o.chain_of_thought for o in outputs],
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)


def run_batch_generation(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    max_vram: int | None,
    callbacks: BaseRunInferenceCallbacks,
    max_output_length: int = 8192,
    batch_size: int | None = None,
) -> None:
    if max_vram is None and batch_size is None:
        max_vram = get_default_device_bytes()
        if max_vram is None:
            raise ValueError(
                "Unable to determine default defice memory capacity; please specify either --vram-gb or --batch-size",
            )

    total_rows = pl.scan_parquet(dataset_path).select(pl.len()).collect().item()

    callbacks.dataset_path = dataset_path
    callbacks.output_path = output_path
    callbacks.max_vram = max_vram
    callbacks.total_rows = total_rows

    callbacks.loading_model()
    model = LanguageModelConfig.load_model(model_path)

    lazy_df = load_hf_parquet(dataset_path)
    collected_df = lazy_df.collect()

    if len(collected_df) == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"response": [], "chain_of_thought": []}).write_parquet(output_path)
        return

    conversations = collected_df.get_column("conversation")
    dataset = iter(
        [HFMessage.from_dict(message).as_message() for message in conversation]
        for conversation in conversations
    )

    inference_config = InferenceConfig(max_output_length=max_output_length, batch_size=batch_size)

    replies: list[tuple[int, AssistantMessage]] = []
    for rows_processed, (idx, reply) in enumerate(
        model.reply_many(
            dataset,
            inference_config=inference_config,
            vram_bytes=max_vram,
            batch_sizes_callback=callbacks.batch_sizes_computed,
        ),
    ):
        replies.append((idx, reply))
        callbacks.generation_progress(rows_processed)

    replies.sort(key=lambda x: x[0])

    df = pl.DataFrame(
        {
            "response": [reply.response for _, reply in replies],
            "chain_of_thought": [reply.chain_of_thought for _, reply in replies],
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
