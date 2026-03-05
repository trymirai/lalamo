import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import requests
from jaxtyping import DTypeLike
from rich import box
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.prompt import Confirm
from rich.table import Table
from typer import Argument, Exit, Option, Typer

from lalamo.cli.common import (
    DEFAULT_OUTPUT_DIR,
    SCRIPT_NAME,
    ModelParser,
    Precision,
    RemoteModelParser,
    _model_size_string_to_int,
    console,
)
from lalamo.common import flatten_parameters
from lalamo.model_import import ModelMetadata, ModelSpec, import_model
from lalamo.model_import.common import (
    DownloadingFileEvent,
    FileSpec,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    StatusEvent,
)
from lalamo.model_import.remote_registry import RegistryModel
from lalamo.model_registry import ModelRegistry
from lalamo.modules import config_converter
from lalamo.safetensors import safe_write

app = Typer()


def _download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def pull(model_spec: RegistryModel, output_dir: Path) -> None:
    if output_dir.exists():
        raise RuntimeError(f"Output directory {output_dir} already exists, refusing to overwrite!")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for file_spec in model_spec.files:
            safe_name = Path(file_spec.name).name
            if not safe_name or safe_name != file_spec.name:
                raise RuntimeError(
                    f"Invalid filename from registry: {file_spec.name!r}. "
                    f"Filenames must not contain path separators or traversal sequences.",
                )

            file_path = temp_path / safe_name
            try:
                _download_file(file_spec.url, file_path)
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download {safe_name}: {e}") from e

            if not file_path.exists():
                raise RuntimeError(f"Downloaded file {safe_name} is missing from {temp_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for file_spec in model_spec.files:
            safe_name = Path(file_spec.name).name
            src = temp_path / safe_name
            dst = output_dir / safe_name
            shutil.move(str(src), str(dst))


@app.command(help="Convert the model for use with the Uzu inference engine.")
def convert(
    model_repo: Annotated[
        ModelSpec,
        Argument(
            help=(
                "Huggingface model repo. Example: [cyan]'meta-llama/Llama-3.2-1B-Instruct'[/cyan]."
                "\n\n\n\n"
                f"You can use the [cyan]`{SCRIPT_NAME} list-models`[/cyan] command to get a list of supported models."
            ),
            click_type=ModelParser(),
            show_default=False,
            metavar="MODEL_REPO",
            autocompletion=lambda: list(ModelRegistry.build().repo_to_model),
        ),
    ],
    precision: Annotated[
        Precision | None,
        Option(
            help="Precision to use for activations and non-quantized weights.",
            show_default="Native precision of the model",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        Option(
            help="Directory to save the converted model to.",
            show_default="Saves the converted model in the `models/<model_name>` directory",
        ),
    ] = None,
    context_length: Annotated[
        int | None,
        Option(
            help="Maximum supported context length. Used to precompute positional embeddings.",
            show_default="Model's native maximum context length.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing model files.",
        ),
    ] = False,
) -> None:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / model_repo.name

    if output_dir.exists():
        if not overwrite and not Confirm().ask(
            rf"⚠️ Output directory [cyan]{output_dir}[/cyan] already exists."
            r" Do you want to overwrite it?",
        ):
            raise Exit
        shutil.rmtree(output_dir)

    precision_dtype: DTypeLike | None = (
        config_converter.structure(precision.value, DTypeLike) if precision is not None else None  # type: ignore[arg-type]
    )

    conversion_strs = [
        f"🚀 Converting [cyan]{model_repo.name}[/cyan] by [cyan]{model_repo.vendor}[/cyan]",
    ]
    if precision is not None:
        conversion_strs.append(
            f" and converting floating-point weights into [cyan]{precision.name.lower()}[/cyan] precision",
        )
    conversion_strs.append(".")
    console.print("".join(conversion_strs))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        downloading_tasks: dict[FileSpec, TaskID] = {}
        initializing_task: TaskID | None = None

        def on_import_progress(event: StatusEvent) -> None:
            nonlocal initializing_task
            match event:
                case DownloadingFileEvent(file_spec):
                    downloading_tasks[file_spec] = progress.add_task(f"Retrieving {file_spec.filename}...")
                case FinishedDownloadingFileEvent(file_spec):
                    progress.remove_task(downloading_tasks.pop(file_spec))
                case InitializingModelEvent():
                    initializing_task = progress.add_task("Initializing model...")
                case FinishedInitializingModelEvent():
                    if initializing_task is not None:
                        progress.remove_task(initializing_task)

        model, metadata = import_model(
            model_repo,
            precision=precision_dtype,
            context_length=context_length,
            progress_callback=on_import_progress,
        )

        saving_task = progress.add_task(f"💾 Saving the model to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        model.message_processor.tokenizer.save(str(output_dir / "tokenizer.json"))
        weights = flatten_parameters(model.export_weights())
        del model

        with Path(output_dir / "model.safetensors").open("wb") as fd:
            safe_write(fd, weights)

        config_json = config_converter.unstructure(metadata, ModelMetadata)
        with open(output_dir / "config.json", "w") as file:
            json.dump(config_json, file, indent=4)

        progress.remove_task(saving_task)

    console.print(f"🧑‍🍳 Model successfully cooked and saved to [cyan]`{output_dir}`[/cyan]!")


@app.command(name="pull", help="Pull a pre-converted model from the SDK repository.")
def pull_command(
    model_spec: Annotated[
        RegistryModel,
        Argument(
            help=(
                "Model repository ID from the pre-converted catalog. "
                "Example: [cyan]'meta-llama/Llama-3.2-1B-Instruct'[/cyan]. "
                "Fuzzy matching is supported for typos and partial names."
            ),
            click_type=RemoteModelParser(),
            show_default=False,
            metavar="MODEL_IDENTIFIER",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        Option(
            help="Directory to save the pulled model to.",
            show_default="Saves the pulled model in the `models/<model_name>` directory",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing model files without prompting.",
        ),
    ] = False,
) -> None:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / model_spec.name

    if output_dir.exists():
        if not overwrite and not Confirm().ask(
            rf"⚠️ Output directory [cyan]{output_dir}[/cyan] already exists."
            r" Do you want to overwrite it?",
        ):
            raise Exit
        shutil.rmtree(output_dir)

    console.print(f"📦 Pulling [cyan]{model_spec.repo_id}[/cyan]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("⬇️  Downloading...")
        pull(model_spec, output_dir)
    console.print(f"🎉 Model successfully pulled to [cyan]{output_dir}[/cyan]!")


@app.command(name="list-models", help="List the supported models.")
def list_models(
    plain: Annotated[
        bool,
        Option(
            help="Only list repo names without fancy formatting.",
        ),
    ] = False,
) -> None:
    registry = ModelRegistry.build()
    sorted_specs = sorted(
        registry.repo_to_model.values(),
        key=lambda spec: (
            spec.vendor.lower(),
            spec.family.lower(),
            _model_size_string_to_int(spec.size),
            spec.name.lower(),
        ),
    )

    if plain:
        for spec in sorted_specs:
            console.print(spec.repo)
        return

    table = Table(
        show_header=True,
        header_style="bold",
        show_lines=True,
        box=box.ROUNDED,
    )
    table.add_column("Vendor", justify="left", style="magenta")
    table.add_column("Family", justify="left", style="magenta", no_wrap=True)
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Quant", justify="left", style="magenta")
    table.add_column("Repo", justify="left", style="cyan", no_wrap=True)
    for spec in sorted_specs:
        table.add_row(
            spec.vendor,
            spec.family,
            spec.size,
            str(spec.quantization),
            spec.repo,
        )
    console.print(table)
