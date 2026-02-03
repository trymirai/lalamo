from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from evals.protocols import EvalAdapter
from evals.types import InferenceOutput, InternalEvalRecord

from lalamo.evals.datasets.specs import EvalSpec
from lalamo.evals.inference.engines import InferenceEngine
from lalamo.models.common import BatchSizesComputedEvent

if TYPE_CHECKING:
    from lalamo.message_processor import AssistantMessage


def run_inference(
    _eval_spec: EvalSpec,
    dataset_dir: Path,
    split: str,
    _model_path: Path,
    output_dir: Path,
    inference_engine: InferenceEngine,
    eval_adapter: EvalAdapter,
    num_few_shot: int = 5,
    max_examples: int | None = None,
    category: str | None = None,
) -> Path:
    """Run model inference on an eval dataset.

    Args:
        eval_spec: Eval specification (e.g., MMLU_PRO)
        dataset_dir: Directory containing converted dataset
        split: Dataset split to use (e.g., "test", "validation")
        model_path: Path to model for inference
        output_dir: Where to save inference results
        inference_engine: Engine to use (lalamo, uzu, llama.cpp, etc.)
        eval_adapter: Eval-specific adapter for prompt formatting
        num_few_shot: Number of few-shot examples
        max_examples: Optional limit on number of examples

    Returns:
        Path to predictions parquet file
    """

    from rich.console import Console

    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test dataset (InternalEvalRecord format)
    console.print("  [cyan]1/6[/cyan] Loading test dataset...")
    test_records = _load_internal_dataset(dataset_dir, split, max_examples, category)
    if category:
        console.print(f"      ✓ Loaded {len(test_records)} test examples (category: {category})")
    else:
        console.print(f"      ✓ Loaded {len(test_records)} test examples")

    # 2. Load validation dataset for few-shot examples
    few_shot_records = None
    if num_few_shot > 0:
        console.print(f"  [cyan]2/6[/cyan] Loading validation dataset for {num_few_shot}-shot examples...")
        few_shot_records = _load_internal_dataset(dataset_dir, "validation", None)
        console.print(f"      ✓ Loaded {len(few_shot_records)} validation examples")
    else:
        console.print("  [cyan]2/6[/cyan] Skipped (0-shot mode)")

    # 3. Format prompts (eval-specific)
    console.print(f"  [cyan]3/6[/cyan] Formatting prompts with {num_few_shot}-shot examples...")
    prompts = eval_adapter.format_prompts(
        records=test_records,
        few_shot_source=few_shot_records,
        num_few_shot=num_few_shot,
    )
    console.print(f"      ✓ Generated {len(prompts)} prompts")

    # 4. Prepare input for engine (engine-specific)
    console.print("  [cyan]4/6[/cyan] Preparing input for inference engine...")
    input_path = output_dir / "inference_input.parquet"
    inference_engine.prepare_input(prompts, input_path)
    console.print(f"      ✓ Saved to {input_path}")

    # 5. Run inference (engine-specific)
    console.print("  [cyan]5/6[/cyan] Running inference (this may take a while)...")
    raw_output_path = output_dir / "inference_output.parquet"
    inference_engine.run_inference(input_path, raw_output_path)
    console.print("      ✓ Inference complete")

    # 6. Parse output (engine-specific)
    console.print("  [cyan]6/6[/cyan] Parsing outputs and saving predictions...")
    outputs = inference_engine.parse_output(raw_output_path, input_path)

    # 7. Save as predictions parquet
    predictions_path = output_dir / f"predictions_{split}.parquet"
    _save_predictions(outputs, predictions_path)
    console.print(f"      ✓ Saved {len(outputs)} predictions")

    return predictions_path


def _load_internal_dataset(
    dataset_dir: Path,
    split: str,
    max_examples: int | None,
    category: str | None = None,
) -> list[InternalEvalRecord]:
    """Load internal format dataset, optionally filtered by category."""
    from evals.types import InternalEvalRecord

    split_file = dataset_dir / f"{split}.parquet"
    df = pl.read_parquet(split_file)

    # Filter by category if specified
    if category:
        df = df.filter(pl.col("category") == category)

    # Limit examples if specified
    if max_examples:
        df = df.head(max_examples)

    records = [InternalEvalRecord(**row) for row in df.iter_rows(named=True)]
    return records


def _save_predictions(
    outputs: list[InferenceOutput],
    output_path: Path,
) -> None:
    """Save inference outputs as predictions parquet."""

    df = pl.DataFrame(
        {
            "id": [o.id for o in outputs],
            "model_output": [o.response for o in outputs],
            "chain_of_thought": [o.chain_of_thought for o in outputs],
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)


@dataclass
class GenerateRepliesCallbacks:
    model_path: Path
    dataset_path: Path
    output_path: Path
    max_vram: int | None
    batch_size: int | None
    total_rows: int

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self) -> None:
        pass

    def estimating_batchsize(self, sequence_length: int, lo: int, hi: int | None) -> None:
        pass

    def batch_sizes_estimated(self) -> None:
        pass

    def batch_sizes_computed(self, event: BatchSizesComputedEvent) -> None:
        pass

    def generation_progress(self, rows_processed: int) -> None:
        pass

    def finished_generation(self) -> None:
        pass


def generate_replies(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    max_vram: int | None,
    max_output_length: int = 8192,
    batch_size: int | None = None,
    callbacks_type: Callable[
        [
            Path,
            Path,
            Path,
            int | None,
            int | None,
            int,
        ],
        GenerateRepliesCallbacks,
    ] = GenerateRepliesCallbacks,
) -> None:
    from lalamo.common import get_default_device_bytes
    from lalamo.data.huggingface_message import import_hf_parquet
    from lalamo.models.common import InferenceConfig
    from lalamo.models.language_model import LanguageModelConfig

    # figure out max_vram if neither batch_size nor max_vram is set
    if max_vram is None and batch_size is None:
        max_vram = get_default_device_bytes()
        if max_vram is None:
            raise ValueError(
                "Unable to determine default defice memory capacity; please specify either --vram-gb or --batch-size",
            )

    # Count rows without loading full dataset
    total_rows = pl.scan_parquet(dataset_path).select(pl.len()).collect().item()

    callbacks = callbacks_type(
        model_path,
        dataset_path,
        output_path,
        max_vram,
        batch_size,
        total_rows,
    )

    callbacks.loading_model()
    model = LanguageModelConfig.load_model(model_path)
    callbacks.finished_loading_model()

    callbacks.loading_dataset()
    dataset = iter(import_hf_parquet(dataset_path, shuffle=False))
    try:
        first_row = next(dataset)
    except StopIteration:
        callbacks.finished_loading_dataset()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"response": [], "chain_of_thought": []}).write_parquet(output_path)
        return
    dataset = chain([first_row], dataset)  # iterator is lazy, force it to actually open the file
    callbacks.finished_loading_dataset()

    inference_config = InferenceConfig(max_output_length=max_output_length, batch_size=batch_size)

    callbacks.batch_sizes_estimated()

    replies: list[tuple[int, AssistantMessage]] = []
    for rows_processed, (idx, reply) in enumerate(
        model.reply_many(
            dataset,
            inference_config,
            vram_bytes=max_vram,
            batch_sizes_callback=callbacks.batch_sizes_computed,
        ),
    ):
        replies.append((idx, reply))
        callbacks.generation_progress(rows_processed)

    # Sort by original index to restore input order
    replies.sort(key=lambda x: x[0])

    df = pl.DataFrame(
        {
            "response": [reply.response for _, reply in replies],
            "chain_of_thought": [reply.chain_of_thought for _, reply in replies],
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    callbacks.finished_generation()
