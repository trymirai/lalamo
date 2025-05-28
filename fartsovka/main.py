import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import thefuzz.process
from click import Context as ClickContext
from click import Parameter as ClickParameter
from click import ParamType
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from safetensors.flax import save_file
from typer import Argument, Exit, Option, Typer

from fartsovka.common import DType
from fartsovka.model_import import REPO_TO_MODEL, ModelSpec, import_model
from fartsovka.modules import DecoderConfig, WeightLayout, config_converter

SCRIPT_NAME = Path(sys.argv[0]).name

DEFAULT_OUTPUT_DIR = Path("models")


class Precision(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


console = Console()
err_console = Console(stderr=True)
app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


class ModelParser(ParamType):
    name: str = "Huggingface Model Repo"

    def convert(self, value: str, param: ClickParameter | None, ctx: ClickContext | None) -> ModelSpec:
        result = REPO_TO_MODEL.get(value)
        if result is None:
            closest_repo = _closest_repo(value)
            error_message_parts = [
                f'"{value}".',
            ]
            if closest_repo:
                error_message_parts.append(
                    f' Perhaps you meant "{closest_repo}"?',
                )
            error_message_parts.append(
                f"\n\nUse the `{SCRIPT_NAME} list-models` command to see the list of currently supported models.",
            )
            error_message = "".join(error_message_parts)
            self.fail(error_message, param, ctx)
        return result


def _closest_repo(query: str, min_score: float = 80) -> str | None:
    if not REPO_TO_MODEL:
        return None
    (closest_match, score), *_ = thefuzz.process.extract(query, list(REPO_TO_MODEL))
    if closest_match and score >= min_score:
        return closest_match
    return None


def _error(message: str) -> None:
    panel = Panel(message, box=box.ROUNDED, title="Error", title_align="left", border_style="red")
    err_console.print(panel)
    raise Exit(1)


@app.command(help="Convert the model for use with the Uzu inference engine.", no_args_is_help=True)
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
            autocompletion=lambda: list(REPO_TO_MODEL),
        ),
    ],
    precision: Annotated[
        Precision | None,
        Option(
            help="Precision to use for activations and non-quantized weights.",
            show_default="Native precision of the model",
        ),
    ] = None,
    weight_layout: Annotated[
        WeightLayout | None,
        Option(
            help="Layout of weights for linear layers.",
            show_default="Output, Input",
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
        int,
        Option(
            help="Maximum supported context length. Used to precompute positional embeddings.",
            min=1,
        ),
    ] = 8192,
) -> None:
    if precision is not None:
        precision_dtype = config_converter.structure(precision.value, DType)  # type: ignore
    else:
        precision_dtype = None

    if weight_layout is not None:
        weight_layout = WeightLayout(weight_layout)
    else:
        weight_layout = WeightLayout.OUTPUT_INPUT

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        console.print(f"🚀 Converting {model_repo.name} by {model_repo.vendor}.")
        conversion_strs = [
            f"⚙️ Using weight layout {weight_layout}",
        ]
        if precision_dtype is not None:
            conversion_strs.append(f" and ({precision_dtype.name}) precision for floating-point weights")
        conversion_strs.append(".")
        console.print("".join(conversion_strs))

        progress.add_task("👨‍🍳 Cooking...")
        model = import_model(model_repo, precision=precision_dtype, context_length=context_length)

        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR / model_repo.name
        progress.add_task(f"💾 Saving model to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        config_json = config_converter.unstructure(model.config, DecoderConfig)
        weights = dict(model.export_weights(weight_layout))
        save_file(weights, output_dir / "model.safetensors")

        with open(output_dir / "config.json", "w") as file:
            json.dump(config_json, file, indent=4)

    console.print("🧑‍🍳 Model successfully cooked!")


@app.command(help="List the supported models.")
def list_models() -> None:
    table = Table(
        show_header=True,
        header_style="bold",
        show_lines=True,
        box=box.ROUNDED,
    )
    table.add_column("Vendor", justify="left", style="magenta")
    table.add_column("Name", justify="left", style="magenta")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Repo", justify="left", style="cyan", no_wrap=True)
    for spec in sorted(REPO_TO_MODEL.values(), key=lambda spec: (spec.vendor.lower(), spec.name.lower())):
        table.add_row(spec.vendor, spec.name, spec.size, spec.repo)
    console.print(table)


if __name__ == "__main__":
    app()
