import json
import re
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import thefuzz.process
from click import Context as ClickContext
from click import Parameter as ClickParameter
from click import ParamType
from jaxtyping import DTypeLike
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from safetensors.flax import save_file
from typer import Argument, Exit, Option, Typer

from lalamo.model_import import REPO_TO_MODEL, ModelMetadata, ModelSpec, import_model
from lalamo.modules import WeightLayout, config_converter
from lalamo.utils import jax_uint4_to_packed_uint8

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


def _pack_uint4_weights(weights: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    packed_weights = {}
    for key, value in weights.items():
        if value.dtype == jnp.uint4:
            packed_weights[key] = jax_uint4_to_packed_uint8(value)
        else:
            packed_weights[key] = value
    return packed_weights


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
            help=(
                "Order of dimensions in the weights of linear layers."
                "\n\n\n\n"
                "If set to AUTO, the layout will depend on the model."
            ),
            show_default="auto",
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
    include_traces: Annotated[
        bool,
        Option(
            help="Export activation traces for debugging purposes.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing model files.",
        ),
    ] = False,
) -> None:
    if precision is not None:
        precision_dtype = config_converter.structure(precision.value, DTypeLike)  # type: ignore
    else:
        precision_dtype = None

    if weight_layout is not None:
        weight_layout = WeightLayout(weight_layout)
    else:
        weight_layout = WeightLayout.AUTO

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / model_repo.name

    console.print(f"🚀 Converting [cyan]{model_repo.name}[/cyan] by [cyan]{model_repo.vendor}[/cyan].")
    conversion_strs = [
        f"⚙️ Using weight layout [cyan]{weight_layout}[/cyan]",
    ]
    if precision is not None:
        conversion_strs.append(
            f" and converting floating-point weights into [cyan]{precision.name.lower()}[/cyan] precision",
        )
    conversion_strs.append(".")
    console.print("".join(conversion_strs))

    if output_dir.exists() and not overwrite:
        answer = console.input(
            rf"⚠️ Output directory [cyan]{output_dir}[/cyan] already exists."
            r" Do you want to overwrite it? [cyan]\[y/n][/cyan]: ",
        )
        while answer.lower() not in ["y", "n", "yes", "no"]:
            answer = console.input("Please enter 'y' or 'n': ")
        if answer.lower() in ["y", "yes"]:
            shutil.rmtree(output_dir)
        else:
            console.print("Exiting...")
            raise Exit

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("👨‍🍳 Cooking...")
        model, metadata, tokenizer_file_paths = import_model(
            model_repo,
            precision=precision_dtype,
            context_length=context_length,
        )
        progress.add_task(f"💾 Saving the model to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        weights = dict(model.export_weights(weight_layout))
        packed_weights = _pack_uint4_weights(weights)
        save_file(packed_weights, output_dir / "model.safetensors")

        config_json = config_converter.unstructure(metadata, ModelMetadata)
        with open(output_dir / "config.json", "w") as file:
            json.dump(config_json, file, indent=4)

        for path in tokenizer_file_paths:
            shutil.copy(path, output_dir / path.name)

        if include_traces:
            progress.add_task("🚁 Generating traces...")

            num_tokens = 512
            token_stride = 8
            token_ids = jnp.arange(0, num_tokens, dtype=jnp.int32)
            token_positions = jnp.arange(0, num_tokens * token_stride, token_stride, dtype=jnp.int32)
            result = model(
                token_ids,
                token_positions,
                return_updated_kv_cache=True,
                return_activation_trace=True,
            )
            traces = dict(result.export())
            save_file(traces, output_dir / "traces.safetensors")

    console.print(f"🧑‍🍳 Model successfully cooked and saved to [cyan]`{output_dir}`[/cyan]!")


def _model_size_string_to_int(
    size_str: str,
    _regex: re.Pattern = re.compile(r"(?P<number>(\d+)(\.\d*)?)(?P<suffix>[KMBT])"),
) -> float:
    match = _regex.match(size_str)
    factors = {
        "K": 1024**1,
        "M": 1024**2,
        "B": 1024**3,
        "T": 1024**4,
    }
    if match:
        return float(match.group("number")) * factors[match.group("suffix")]
    raise ValueError(f"Invalid size string: {size_str}")


@app.command(help="List the supported models.")
def list_models(
    plain: Annotated[
        bool,
        Option(
            help="Only list repo names without fancy formatting.",
        ),
    ] = False,
) -> None:
    sorted_specs = sorted(
        REPO_TO_MODEL.values(),
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


if __name__ == "__main__":
    app()
