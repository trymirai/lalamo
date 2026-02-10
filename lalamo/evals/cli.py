from pathlib import Path
from typing import Annotated

from openai import OpenAIError
from rich.console import Console
from typer import Argument, Exit, Option, Typer

from lalamo.evals.benchmark.callbacks import ConsoleCallbacks as BenchmarkConsoleCallbacks
from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.datasets.callbacks import ConsoleCallbacks as DatasetConsoleCallbacks
from lalamo.evals.datasets.command import convert_dataset_handler
from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference.callbacks import ConsoleRunInferenceCallbacks
from lalamo.evals.inference.command import infer_command_handler
from lalamo.evals.inference.engines import CustomAPIEngineConfig, LalamoEngineConfig

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
    dataset_dir: Annotated[Path, Argument(help="Path to converted dataset directory")],
    output_dir: Annotated[Path, Argument(help="Output directory for results")],
    # Engine selection
    engine: Annotated[
        str,
        Option(help="Inference engine: lalamo, custom_api"),
    ] = "lalamo",
    limit: Annotated[int | None, Option(help="Limit number of test examples to run")] = None,
    # Lalamo engine parameters
    model_path: Annotated[
        Path | None,
        Option(help="[Lalamo] Path to converted model"),
    ] = None,
    batch_size: Annotated[
        int | None,
        Option(help="[Lalamo] Fixed batch size, skipping automatic estimation"),
    ] = None,
    vram_gb: Annotated[
        int | None,
        Option(help="[Lalamo] Maximum VRAM in GB (auto-detect if not set)"),
    ] = None,
    # API engine parameters
    api_base_url: Annotated[
        str | None,
        Option(help="[API] Base URL (e.g., http://localhost:11434/v1)"),
    ] = None,
    api_model: Annotated[
        str | None,
        Option(help="[API] Model name (e.g., qwen2.5-coder:0.5b-instruct)"),
    ] = None,
    api_key: Annotated[
        str | None,
        Option(help="[API] API key (optional)", envvar="OPENAI_API_KEY"),
    ] = None,
    max_retries: Annotated[
        int,
        Option(help="[API] Max retries for failed requests"),
    ] = 0,
    timeout: Annotated[
        float,
        Option(help="[API] Request timeout in seconds"),
    ] = 60.0,
    # Inference config overrides (use adapter's reference values if not set)
    temperature: Annotated[
        float | None,
        Option(help="Sampling temperature (uses adapter default if not set)"),
    ] = None,
    max_output_length: Annotated[
        int | None,
        Option(help="Max tokens to generate (uses adapter default if not set)"),
    ] = None,
    max_model_len: Annotated[
        int | None,
        Option(help="Max sequence length (uses adapter default if not set)"),
    ] = None,
    top_p: Annotated[
        float | None,
        Option(help="Nucleus sampling top_p (uses adapter default if not set)"),
    ] = None,
    top_k: Annotated[
        int | None,
        Option(help="Top-k sampling (uses adapter default if not set)"),
    ] = None,
    stop_tokens: Annotated[
        list[str] | None,
        Option(help="Stop token strings (uses adapter default if not set)"),
    ] = None,
) -> None:
    eval_spec = REPO_TO_EVAL.get(eval_repo)
    if eval_spec is None:
        available = ", ".join(REPO_TO_EVAL.keys())
        err_console.print(f"[red]✗[/red] Unknown eval repository: {eval_repo}. Available evals: {available}")
        raise Exit(1)

    if engine == "lalamo":
        if model_path is None:
            err_console.print("[red]✗[/red] --model-path required for --engine lalamo")
            raise Exit(1)

        engine_config = LalamoEngineConfig(
            model_path=model_path,
            batch_size=batch_size,
            vram_gb=vram_gb,
        )

    elif engine == "custom_api":
        if api_base_url is None:
            err_console.print("[red]✗[/red] --api-base-url required for --engine custom_api")
            raise Exit(1)
        if api_model is None:
            err_console.print("[red]✗[/red] --api-model required for --engine custom_api")
            raise Exit(1)

        if batch_size is not None or vram_gb is not None:
            err_console.print(
                "[yellow]⚠[/yellow] --batch-size and --vram-gb are ignored for --engine custom_api",
            )

        engine_config = CustomAPIEngineConfig(
            base_url=api_base_url,
            model=api_model,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

    else:
        err_console.print(f"[red]✗[/red] Unknown engine: {engine}. Supported: lalamo, custom_api")
        raise Exit(1)


    inference_overrides = {
        "temperature": temperature,
        "max_output_length": max_output_length,
        "max_model_len": max_model_len,
        "top_p": top_p,
        "top_k": top_k,
        "stop_tokens": stop_tokens,
    }

    try:
        _predictions_path = infer_command_handler(
            eval_repo=eval_repo,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            engine_config=engine_config,
            inference_overrides=inference_overrides,
            limit=limit,
            callbacks=ConsoleRunInferenceCallbacks(
                eval_repo=eval_repo,
                model_path=model_path,
                limit=limit,
                engine_type=engine,
                engine_config=engine_config,
            ),
        )
    except (ValueError, FileNotFoundError, OpenAIError) as e:
        err_console.print("")
        err_console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None


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
        err_console.print(f"[red]✗[/red] {e}")
        raise Exit(1) from None

