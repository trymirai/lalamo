from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Argument, Exit, Option, Typer

from lalamo.evals.benchmark.callbacks import ConsoleCallbacks as BenchmarkConsoleCallbacks
from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.datasets.callbacks import ConsoleCallbacks as DatasetConsoleCallbacks
from lalamo.evals.datasets.command import convert_dataset_handler
from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference.callbacks import ConsoleRunInferenceCallbacks
from lalamo.evals.inference.command import infer_command_handler

console = Console()
eval_app = Typer()


@eval_app.command(
    name="convert-dataset",
    help="Download and convert evaluation dataset.",
)
def convert_dataset_command(
    eval_name: Annotated[
        str,
        Argument(
            help="Eval name. Example: [cyan]'MMLU-Pro'[/cyan].",
            metavar="EVAL_NAME",
            autocompletion=lambda: list(REPO_TO_EVAL.keys()),
        ),
    ],
    output_dir: Annotated[
        Path | None,
        Option(
            help="Directory to save the dataset to.",
            show_default="Saves the dataset in the `evals/datasets/<name>` directory (using eval's short name)",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing dataset files.",
        ),
    ] = False,
) -> None:
    if eval_name not in REPO_TO_EVAL:
        available = ", ".join(REPO_TO_EVAL.keys())
        console.print(f"[red]✗[/red] Unknown eval: {eval_name}. Available evals: {available}")
        raise Exit(1)

    eval_spec = REPO_TO_EVAL[eval_name]

    if output_dir is None:
        output_dir = Path("evals/datasets") / eval_spec.name

    try:
        convert_dataset_handler(
            eval_name=eval_name,
            output_dir=output_dir,
            callbacks=DatasetConsoleCallbacks(eval_name=eval_name, output_dir=output_dir, overwrite=overwrite),
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None


@eval_app.command(
    name="infer",
    help="Run model inference on evaluation dataset.",
)
def infer_command(
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
    try:
        _predictions_path = infer_command_handler(
            eval_name=eval_name,
            model_path=model_path,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            callbacks=ConsoleRunInferenceCallbacks(
                eval_name=eval_name,
                model_path=model_path,
                split=split,
                num_few_shot=num_few_shot,
                category=category,
                max_examples=max_examples,
                batch_size=batch_size,
                vram_gb=vram_gb,
            ),
            split=split,
            engine=engine,
            num_few_shot=num_few_shot,
            max_examples=max_examples,
            category=category,
            batch_size=batch_size,
            vram_gb=vram_gb,
            max_output_length=max_output_length,
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Inference interrupted by user")
        raise Exit(130) from None


@eval_app.command(
    name="benchmark",
    help="Run benchmark evaluation on model predictions.",
)
def benchmark_command(
    eval_name: Annotated[str, Argument(help="Eval name (e.g., MMLU-Pro)")],
    predictions_path: Annotated[Path, Argument(help="Path to predictions parquet file")],
    dataset_dir: Annotated[Path, Argument(help="Path to converted dataset directory")],
    split: Annotated[str, Option(help="Dataset split to use")] = "test",
) -> None:
    try:
        benchmark_command_handler(
            eval_name=eval_name,
            predictions_path=predictions_path,
            dataset_dir=dataset_dir,
            callbacks=BenchmarkConsoleCallbacks(),
            split=split,
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None

