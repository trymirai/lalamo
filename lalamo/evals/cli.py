from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Argument, Exit, Option, Typer

from lalamo.common import vram_gb_to_bytes
from lalamo.evals.benchmark.callbacks import ConsoleCallbacks as BenchmarkConsoleCallbacks
from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.datasets.callbacks import ConsoleCallbacks as DatasetConsoleCallbacks
from lalamo.evals.datasets.command import convert_dataset_handler
from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference.callbacks import ConsoleRunInferenceCallbacks
from lalamo.evals.inference.command import InferenceConfigOverrides, infer_command_handler

console = Console()
err_console = Console(stderr=True)
eval_app = Typer()


@eval_app.command(
    name="convert-dataset",
    help="Download and convert evaluation dataset.",
)
def convert_dataset_command(
    eval_repo: Annotated[
        str,
        Argument(
            help="Eval repository. Example: [cyan]'TIGER-Lab/MMLU-Pro'[/cyan].",
            autocompletion=lambda: list(REPO_TO_EVAL.keys()),
        ),
    ],
    output_dir: Annotated[
        Path | None,
        Option(
            help="Directory to save the dataset to.",
            show_default="Saves the dataset in the `data/evals/datasets/<name>` directory (using eval's short name)",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing dataset files.",
        ),
    ] = False,
) -> None:
    eval_spec = REPO_TO_EVAL.get(eval_repo)
    if eval_spec is None:
        available = ", ".join(REPO_TO_EVAL.keys())
        err_console.print(f"[red]✗[/red] Unknown eval repository: {eval_repo}. Available evals: {available}")
        raise Exit(1)

    if output_dir is None:
        output_dir = Path("data/evals/datasets") / eval_spec.name

    try:
        convert_dataset_handler(
            eval_repo=eval_repo,
            output_dir=output_dir,
            callbacks=DatasetConsoleCallbacks(eval_repo=eval_repo, output_dir=output_dir, overwrite=overwrite),
        )
    except ValueError as e:
        err_console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None


@eval_app.command(
    name="infer",
    help="Run model inference on evaluation dataset.",
)
def infer_command(
    eval_repo: Annotated[
        str,
        Argument(
            help="Eval repository. Example: [cyan]'TIGER-Lab/MMLU-Pro'[/cyan].",
            autocompletion=lambda: list(REPO_TO_EVAL.keys()),
        ),
    ],
    model_path: Annotated[Path, Argument(help="Path to converted model")],
    dataset_dir: Annotated[Path, Argument(help="Path to converted dataset directory")],
    output_dir: Annotated[Path, Argument(help="Output directory for results")],
    engine: Annotated[str, Option(help="Inference engine")] = "lalamo",
    limit: Annotated[int | None, Option(help="Limit number of test examples to run")] = None,
    batch_size: Annotated[
        int | None,
        Option(help="Fixed batch size to use, skipping automatic estimation."),
    ] = None,
    vram_gb: Annotated[
        int | None,
        Option(
            help="Maximum VRAM in GB. Batch sizes are estimated automatically.",
            show_default="max on default device",
        ),
    ] = None,
    # Inference config overrides (use adapter's reference values if not set)
    temperature: Annotated[
        float | None,
        Option(help="Sampling temperature (uses adapter's reference value if not set)"),
    ] = None,
    max_output_length: Annotated[
        int | None,
        Option(help="Max tokens to generate per response (uses adapter's reference value if not set)"),
    ] = None,
    max_model_len: Annotated[
        int | None,
        Option(help="Max total sequence length (uses adapter's reference value if not set)"),
    ] = None,
    top_p: Annotated[
        float | None,
        Option(help="Nucleus sampling top_p (uses adapter's reference value if not set)"),
    ] = None,
    top_k: Annotated[
        int | None,
        Option(help="Top-k sampling (uses adapter's reference value if not set)"),
    ] = None,
    stop_tokens: Annotated[
        list[str] | None,
        Option(help="Stop token strings (uses adapter's reference value if not set)"),
    ] = None,
) -> None:
    if batch_size is not None and vram_gb is not None:
        err_console.print("Cannot use both --batch-size and --vram-gb")
        raise Exit(1)

    max_vram: int | None = None
    if batch_size is None:
        max_vram = vram_gb_to_bytes(vram_gb)
        if max_vram is None:
            err_console.print("Cannot get the default device's memory stats, use --vram-gb or --batch-size")
            raise Exit(1)

    eval_spec = REPO_TO_EVAL.get(eval_repo)
    if eval_spec is None:
        available = ", ".join(REPO_TO_EVAL.keys())
        err_console.print(f"[red]✗[/red] Unknown eval repository: {eval_repo}. Available evals: {available}")
        raise Exit(1)

    inference_overrides = InferenceConfigOverrides(
        temperature=temperature,
        max_output_length=max_output_length,
        max_model_len=max_model_len,
        top_p=top_p,
        top_k=top_k,
        stop_tokens=stop_tokens,
    )

    try:
        _predictions_path = infer_command_handler(
            eval_repo=eval_repo,
            model_path=model_path,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            callbacks=ConsoleRunInferenceCallbacks(
                eval_repo=eval_repo,
                model_path=model_path,
                limit=limit,
                batch_size=batch_size,
                vram_gb=vram_gb,
            ),
            inference_overrides=inference_overrides,
            limit=limit,
            batch_size=batch_size,
            max_vram_bytes=max_vram,
            engine=engine,
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None
    except FileNotFoundError as e:
        # Extract the path from the error message
        error_msg = str(e)
        if "config.json" in error_msg:
            console.print(f"[red]✗[/red] Model not found: {model_path}")
            console.print("    Make sure the model directory exists and contains config.json")
        elif ".parquet" in error_msg:
            # Extract file path from polars error message
            if ": " in error_msg and "\n" in error_msg:
                file_path = error_msg.split(": ", 1)[1].split("\n", 1)[0]
            else:
                file_path = "dataset file"
            console.print(f"[red]✗[/red] Dataset not found: {file_path}")
            console.print(f"    Run: lalamo eval convert-dataset {eval_repo}")
        else:
            console.print(f"[red]✗[/red] File not found: {e}")
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
) -> None:
    try:
        benchmark_command_handler(
            eval_name=eval_name,
            predictions_path=predictions_path,
            callbacks=BenchmarkConsoleCallbacks(),
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None

