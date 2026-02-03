from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Argument, Exit, Option, Typer

from lalamo.evals.benchmark.callbacks import ConsoleCallbacks as BenchmarkConsoleCallbacks
from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.datasets.callbacks import ConsoleCallbacks as DatasetConsoleCallbacks
from lalamo.evals.datasets.command import convert_dataset_handler
from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference.cli import infer_command

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
            show_default="Saves the dataset in the `datasets/<eval_name>` directory",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing dataset files.",
        ),
    ] = False,
) -> None:
    if output_dir is None:
        output_dir = Path("datasets") / eval_name

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


eval_app.command(
    name="infer",
    help="Run model inference on evaluation dataset.",
)(infer_command)
