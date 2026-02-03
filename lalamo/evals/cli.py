from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Annotated

import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.prompt import Confirm
from typer import Argument, Exit, Option, Typer

from lalamo.evals.datasets import REPO_TO_EVAL, EvalConversionCallbacks, EvalSpec, convert_dataset

console = Console()
eval_app = Typer()

DEFAULT_DATASETS_DIR = Path("datasets")


@dataclass
class CliEvalConversionCallbacks(EvalConversionCallbacks):
    overwrite: bool = False

    stack: ExitStack = field(default_factory=ExitStack)
    progress: Progress | None = None
    downloading_tasks: dict[str, TaskID] = field(default_factory=dict)
    converting_task: TaskID | None = None

    def started(self) -> None:
        console.print(f"🚀 Converting eval dataset [cyan]{self.eval_spec.name}[/cyan].")

        self.progress = self.stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ),
        )

    def output_dir_exists(self) -> None:
        if not self.overwrite and not Confirm().ask(
            rf"⚠️ Output directory [cyan]{self.output_dir}[/cyan] already exists. Continue?",
        ):
            raise Exit(0)

    def downloading_file(self, filename: str) -> None:
        assert self.progress is not None
        self.downloading_tasks[filename] = self.progress.add_task(f"Retrieving {filename}...")

    def finished_downloading_file(self, filename: str) -> None:
        assert self.progress is not None
        self.progress.remove_task(self.downloading_tasks[filename])

    def converting_split(self, split: str) -> None:
        assert self.progress is not None
        self.converting_task = self.progress.add_task(f"Converting {split} split to internal format...")

    def finished_converting_split(self, split: str) -> None:
        assert self.progress is not None
        assert self.converting_task is not None
        self.progress.remove_task(self.converting_task)

    def saving_dataset(self) -> None:
        pass

    def finished(self) -> None:
        if self.progress is not None:
            self.stack.close()
        console.print(f"✅ Dataset converted successfully to [cyan]{self.output_dir}[/cyan]")


# Import EvalParser from main.py - need to copy this too
from click import Context as ClickContext
from click import Parameter as ClickParameter
from click.types import ParamType


class EvalParser(ParamType):
    name: str = "Huggingface Eval Repo"

    def convert(self, value: str, param: ClickParameter | None, ctx: ClickContext | None) -> EvalSpec:
        result = REPO_TO_EVAL.get(value)
        if result is None:
            from difflib import get_close_matches
            close_matches = get_close_matches(value, REPO_TO_EVAL.keys(), n=1, cutoff=0.6)
            closest_repo = close_matches[0] if close_matches else None
            error_message_parts = [
                f'"{value}".',
            ]
            if closest_repo:
                error_message_parts.append(
                    f' Perhaps you meant "{closest_repo}"?',
                )
            error_message = "".join(error_message_parts)
            return self.fail(error_message, param, ctx)
        return result


@eval_app.command(name="convert-dataset", help="Download and convert evaluation dataset.")
def convert_dataset_command(
    eval_repo: Annotated[
        EvalSpec,
        Argument(
            help=(
                "HuggingFace eval repo. Example: [cyan]'TIGER-Lab/MMLU-Pro'[/cyan]."
            ),
            click_type=EvalParser(),
            show_default=False,
            metavar="EVAL_REPO",
            autocompletion=lambda: list(REPO_TO_EVAL),
        ),
    ],
    output_dir: Annotated[
        Path | None,
        Option(
            help="Directory to save the dataset to.",
            show_default="Saves the dataset in the `datasets/<eval_name>` directory",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing dataset files.",
        ),
    ] = False,
    max_examples: Annotated[
        int | None,
        Option(
            help="Maximum number of examples to convert per split (for testing).",
            show_default="Convert all examples",
        ),
    ] = None,
) -> None:
    if output_dir is None:
        output_dir = DEFAULT_DATASETS_DIR / eval_repo.name

    convert_dataset(
        eval_repo,
        output_dir,
        partial(CliEvalConversionCallbacks, overwrite=overwrite),
        max_examples=max_examples,
    )


@eval_app.command(name="infer", help="Run model inference on evaluation dataset.")
def infer(
    eval_name: Annotated[str, Argument(help="Eval name (e.g., MMLU-Pro)")],
    model_path: Annotated[Path, Argument(help="Path to converted model")],
    dataset_dir: Annotated[Path, Argument(help="Path to converted dataset directory")],
    output_dir: Annotated[Path, Argument(help="Output directory for results")],
    split: Annotated[str, Option(help="Dataset split to use")] = "test",
    engine: Annotated[str, Option(help="Inference engine")] = "lalamo",
    num_few_shot: Annotated[
        int,
        Option("-k", "--num-few-shot", help="Number of few-shot examples (k-shot learning)"),
    ] = 5,
    max_examples: Annotated[int | None, Option(help="Limit number of examples")] = None,
    category: Annotated[str | None, Option(help="Filter to specific category (e.g., 'business', 'math')")] = None,
    batch_size: Annotated[
        int | None, Option("--batch-size", help="Batch size for inference (auto-computed if not set)"),
    ] = None,
    vram_gb: Annotated[
        float | None, Option(help="VRAM limit in GB (auto-detected if not set)"),
    ] = None,
    max_output_length: Annotated[int, Option(help="Max tokens to generate per response")] = 2048,
) -> None:
    """Run model inference on an eval dataset.

    The -k/--num-few-shot parameter controls k-shot learning:
    - k=0: Zero-shot (no examples, just the question)
    - k=5: Five-shot (5 examples with answers, then the question)
    - k=N: N-shot (N examples with answers, then the question)

    Examples:
        # Run MMLU-Pro with 5-shot prompts, auto batch size
        lalamo eval infer MMLU-Pro models/llama-3.2-1B datasets/MMLU-Pro results/

        # Run with 0-shot (direct answering, no examples)
        lalamo eval infer MMLU-Pro models/llama-3.2-1B datasets/MMLU-Pro results/ -k 0

        # Run only on business category
        lalamo eval infer MMLU-Pro models/llama-3.2-1B datasets/MMLU-Pro results/ --category business

        # Limit to 100 examples from math category
        lalamo eval infer MMLU-Pro models/llama-3.2-1B datasets/MMLU-Pro results/ --category math --max-examples 100
    """
    from lalamo.evals.datasets.specs import REPO_TO_EVAL
    from lalamo.evals.inference import LalamoInferenceEngine
    from lalamo.evals.inference import run_inference as run_eval_inference

    # Load eval spec
    if eval_name not in REPO_TO_EVAL:
        console.print(f"[red]✗[/red] Unknown eval: {eval_name}")
        console.print(f"Available evals: {', '.join(REPO_TO_EVAL.keys())}")
        raise Exit(1)

    eval_spec = REPO_TO_EVAL[eval_name]

    # Create inference engine
    if engine == "lalamo":
        inference_engine = LalamoInferenceEngine(
            model_path=model_path,
            max_vram=int(vram_gb * 1024**3) if vram_gb else None,
            batch_size=batch_size,
            max_output_length=max_output_length,
        )
    else:
        console.print(f"[red]✗[/red] Unsupported engine: {engine}. Supported: lalamo")
        raise Exit(1)

    # Load eval adapter
    eval_adapter = eval_spec.handler_type()

    # Run inference
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Eval: {eval_name}")
    console.print(f"  Model: {model_path}")
    console.print(f"  Split: {split}")
    console.print(f"  Few-shot (k): {num_few_shot}")
    console.print(f"  Batch size: {batch_size or 'auto'}")
    console.print(f"  VRAM limit: {f'{vram_gb} GB' if vram_gb else 'auto-detect'}")
    console.print(f"  Category: {category or 'all'}")
    console.print(f"  Max examples: {max_examples or 'all'}")
    console.print()

    # Run inference with detailed progress
    console.print(f"[bold]Running {num_few_shot}-shot inference...[/bold]")
    try:
        predictions_path = run_eval_inference(
            eval_spec=eval_spec,
            dataset_dir=dataset_dir,
            split=split,
            model_path=model_path,
            output_dir=output_dir,
            inference_engine=inference_engine,
            eval_adapter=eval_adapter,
            num_few_shot=num_few_shot,
            max_examples=max_examples,
            category=category,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Inference interrupted by user")
        raise Exit(130)  # Standard exit code for CTRL-C

    console.print()
    console.print(f"[green]✓[/green] Predictions saved to: {predictions_path}")
    console.print("\nNext steps:")
    console.print(f"  lalamo eval benchmark {eval_name} {predictions_path} {dataset_dir}")


@eval_app.command(name="benchmark", help="Run benchmark evaluation on model predictions.")
def benchmark(
    eval_name: Annotated[str, Argument(help="Eval name (e.g., MMLU-Pro)")],
    predictions_path: Annotated[Path, Argument(help="Path to predictions parquet file")],
    dataset_dir: Annotated[Path, Argument(help="Path to converted dataset directory")],
    split: Annotated[str, Option(help="Dataset split to use")] = "test",
    model_name: Annotated[str | None, Option(help="Model name for results")] = None,
) -> None:
    """Run benchmark evaluation using official eval code.

    This command:
    1. Loads predictions and ground truth
    2. Converts to format expected by official eval code
    3. Runs official benchmark (exact match with reference)
    4. Reports accuracy and metrics

    Examples:
        # Run MMLU-Pro benchmark
        lalamo eval benchmark MMLU-Pro results/predictions_test.parquet datasets/MMLU-Pro

        # With custom model name
        lalamo eval benchmark MMLU-Pro results/predictions.parquet datasets/MMLU-Pro --model-name "my-model"
    """
    from evals.types import InternalEvalRecord, PredictionRecord

    from lalamo.evals.datasets.specs import REPO_TO_EVAL

    # Load eval spec
    if eval_name not in REPO_TO_EVAL:
        console.print(f"[red]✗[/red] Unknown eval: {eval_name}")
        console.print(f"Available evals: {', '.join(REPO_TO_EVAL.keys())}")
        raise Exit(1)

    eval_spec = REPO_TO_EVAL[eval_name]
    eval_adapter = eval_spec.handler_type()

    # Derive model name from path if not provided
    if model_name is None:
        model_name = predictions_path.parent.name or "unknown"

    console.print("[bold]Benchmark Configuration:[/bold]")
    console.print(f"  Eval: {eval_spec.name}")
    console.print(f"  Model: {model_name}")
    console.print(f"  Split: {split}")
    console.print(f"  Predictions: {predictions_path}")
    console.print()

    # 1. Load predictions
    with console.status("[bold green]Loading predictions..."):
        pred_df = pl.read_parquet(predictions_path)
        predictions = []
        for row in pred_df.iter_rows(named=True):
            predictions.append(PredictionRecord(
                id=row["id"],
                model_output=row["model_output"],
            ))
        console.print(f"✓ Loaded {len(predictions)} predictions")

    # 2. Load ground truth (filtered to matching predictions)
    with console.status("[bold green]Loading ground truth..."):
        gt_path = dataset_dir / f"{split}.parquet"
        if not gt_path.exists():
            console.print(f"[red]✗[/red] Ground truth not found: {gt_path}")
            raise Exit(1)

        gt_df = pl.read_parquet(gt_path)

        # Filter to only records that have predictions
        prediction_ids = {p.id for p in predictions}
        ground_truth = []
        for row in gt_df.iter_rows(named=True):
            if row["id"] in prediction_ids:
                ground_truth.append(InternalEvalRecord(**row))

        console.print(f"✓ Loaded {len(ground_truth)} ground truth records (matched to predictions)")

        if len(ground_truth) != len(predictions):
            console.print(f"[yellow]⚠[/yellow] Mismatch: {len(predictions)} predictions but {len(ground_truth)} ground truth")
            console.print("[yellow]⚠[/yellow] Some prediction IDs may not exist in ground truth")

        # Sort both by ID to ensure matching order
        predictions.sort(key=lambda p: p.id)
        ground_truth.sort(key=lambda g: g.id)

    # 3. Prepare for benchmark (convert to official format)
    with console.status("[bold green]Preparing for benchmark..."):
        output_dir = predictions_path.parent / "benchmark_data"
        prepared_path = eval_adapter.prepare_for_benchmark(
            predictions=predictions,
            ground_truth=ground_truth,
            output_dir=output_dir,
        )
        console.print(f"✓ Prepared benchmark data at {prepared_path}")

    # 4. Run official benchmark
    with console.status("[bold green]Running benchmark..."):
        metrics = eval_adapter.run_benchmark(
            prepared_data_path=prepared_path,
            eval_name=eval_spec.name,
            model_name=model_name,
            split=split,
        )
        console.print("✓ Benchmark complete")

    # 5. Display results
    console.print()
    console.print("[bold]═══ Benchmark Results ═══[/bold]")
    console.print(f"Eval: {metrics.eval_name}")
    console.print(f"Model: {metrics.model_name}")
    console.print(f"Split: {metrics.split}")
    console.print()
    console.print(f"[bold green]Overall Accuracy: {metrics.overall_accuracy:.2%}[/bold green]")
    console.print(f"Correct: {metrics.correct}/{metrics.total_examples}")
    console.print(f"Incorrect: {metrics.incorrect}/{metrics.total_examples}")

    if metrics.category_metrics:
        console.print()
        console.print("[bold]Category Breakdown:[/bold]")
        for category, accuracy in sorted(metrics.category_metrics.items()):
            console.print(f"  {category:20s} {accuracy:.2%}")

    if metrics.custom_metrics:
        console.print()
        console.print("[bold]Custom Metrics:[/bold]")
        for metric_name, value in metrics.custom_metrics.items():
            console.print(f"  {metric_name}: {value}")
