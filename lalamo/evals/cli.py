from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Argument, Exit, Option, Typer

from lalamo.evals.benchmark.callbacks import ConsoleCallbacks
from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.datasets.cli import convert_dataset_command
from lalamo.evals.inference.cli import infer_command

console = Console()
eval_app = Typer()

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
            callbacks=ConsoleCallbacks(),
            split=split,
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None


eval_app.command(
    name="convert-dataset",
    help="Download and convert evaluation dataset.",
)(convert_dataset_command)

eval_app.command(
    name="infer",
    help="Run model inference on evaluation dataset.",
)(infer_command)
