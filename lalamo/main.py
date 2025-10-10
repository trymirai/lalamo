import json
import random
import re
import shutil
import sys
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import jax.profiler
import thefuzz.process
from click import Context as ClickContext
from click import Parameter as ClickParameter
from click import ParamType
from jaxtyping import DTypeLike
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from rich.table import Table
from safetensors.flax import save_file
from typer import Argument, Context, Exit, Option, Typer

from lalamo.common import flatten_parameters
from lalamo.data import import_hf_parquet
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.language_model import LanguageModel
from lalamo.message_processor import UserMessage
from lalamo.model_import import REPO_TO_MODEL, ModelMetadata, ModelSpec, import_model
from lalamo.model_import.common import (
    DownloadingFileEvent,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    StatusEvent,
)
from lalamo.modules import config_converter
from lalamo.speculator.inference import CollectTracesEvent, inference_collect_traces
from lalamo.speculator.ngram import NGramSpeculator
from lalamo.speculator.utils import SpeculatorTrainingEvent, test_speculator, train_speculator
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


@app.command(help="Chat with a converted model.")
def chat(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory.",
            metavar="MODEL_PATH",
        ),
    ],
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        loading_task = progress.add_task("🚀 [cyan]Loading model...[/cyan]")
        model = LanguageModel.load(model_path)
        progress.remove_task(loading_task)
        warmup_task = progress.add_task("🔥 Warming up compilation cache...")
        list(model.stream_reply_text([UserMessage("")], max_output_length=1))
        progress.remove_task(warmup_task)
    console.print(f"🤖 Chatting with [blue]{model_path}[/blue]:")
    messages = []
    while True:
        user_text = console.input("[cyan]user> [/cyan]")
        user_message = UserMessage(user_text)
        messages.append(user_message)

        console.print("[red]assistant> [/red]", end="")
        model_response_tokens = []
        for token in model.stream_reply_text(messages):
            console.print(token, end="")
            model_response_tokens.append(token)
        console.print()
        model_response_text = "".join(model_response_tokens)
        messages.append(model.message_processor.parse_response(model_response_text))


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

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / model_repo.name

    conversion_strs = [f"🚀 Converting [cyan]{model_repo.name}[/cyan] by [cyan]{model_repo.vendor}[/cyan]"]
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
        event_to_task = {}

        def progress_callback(event: StatusEvent) -> None:
            match event:
                case DownloadingFileEvent(file_spec):
                    event_to_task[event] = progress.add_task(f"Retrieving {file_spec.filename}...")
                case FinishedDownloadingFileEvent(file_spec):
                    progress.remove_task(event_to_task[event])
                case InitializingModelEvent():
                    event_to_task[event] = progress.add_task("Initializing model...")
                case FinishedInitializingModelEvent():
                    progress.remove_task(event_to_task[event])

        main_task = progress.add_task("👨‍🍳 Cooking...")
        model, metadata = import_model(
            model_repo,
            precision=precision_dtype,
            context_length=context_length,
            progress_callback=progress_callback,
        )
        save_task = progress.add_task(f"💾 Saving the model to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        if include_traces:
            trace_task = progress.add_task("🚁 Generating traces...")

            num_tokens = 512
            token_stride = 8
            token_ids = jnp.arange(0, num_tokens, dtype=jnp.int32)
            token_positions = jnp.arange(0, num_tokens * token_stride, token_stride, dtype=jnp.int32)
            result = model.decoder(
                token_ids,
                token_positions,
                return_updated_kv_cache=True,
                return_activation_trace=True,
            )
            traces = flatten_parameters(result.export())
            save_file(traces, output_dir / "traces.safetensors")
            progress.remove_task(trace_task)
        progress.remove_task(main_task)

        model.message_processor.tokenizer.save(str(output_dir / "tokenizer.json"))
        weights = flatten_parameters(model.export_weights())
        del model

        packed_weights = _pack_uint4_weights(weights)
        save_file(packed_weights, output_dir / "model.safetensors")

        config_json = config_converter.unstructure(metadata, ModelMetadata)
        with open(output_dir / "config.json", "w") as file:
            json.dump(config_json, file, indent=4)
        progress.remove_task(save_task)

    console.print(f"🧑‍🍳 Model successfully cooked and saved to [cyan]`{output_dir}`[/cyan]!")


def _model_size_string_to_int(
    size_str: str,
    _regex: re.Pattern = re.compile(r"(?P<number>(\d+)(\.\d*)?)(?P<suffix>[KMBT])"),
) -> float:
    match = _regex.match(size_str)
    factors = {
        "K": 1000**1,
        "M": 1000**2,
        "B": 1000**3,
        "T": 1000**4,
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


speculator_app = Typer()
app.add_typer(speculator_app, name="speculator", help="Train a speculator for a model.")


@speculator_app.command(help="Run model inference and collect traces for speculator training")
def collect_traces(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory",
            metavar="MODEL_PATH",
        ),
    ],
    dataset_path: Annotated[
        Path,
        Argument(
            help="Path to the dataset with prompts",
            metavar="DATASET_PATH",
        ),
    ],
    output_path: Annotated[
        Path,
        Option(
            help="File to save the trace to",
            metavar="OUTPUT_PATH",
        ),
    ],
    num_logits_per_token: Annotated[
        int,
        Option(help="Record logits for this number of most probable tokens"),
    ] = 8,
    max_input_length: Annotated[
        int,
        Option(help="Filter prompts that have more than this number of tokens in context"),
    ] = 1024,
    max_output_length: Annotated[
        int,
        Option(help="Maximum number of tokens to generate in one completion"),
    ] = 1024,
    batch_size: Annotated[
        int,
        Option(help="Number of sequences in one batch"),
    ] = 1,
    num_tokens_to_generate: Annotated[
        int | None,
        Option(
            help="Exit early after generating this number of output tokens",
            show_default="all",
        ),
    ] = None,
) -> None:
    with Live(refresh_per_second=10) as live:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            disable=True,
        ) as progress:
            live.update(progress, refresh=True)
            loading_model_task = progress.add_task("🧠 [cyan]Loading model...[/cyan]")
            model = LanguageModel.load(model_path)
            progress.remove_task(loading_model_task)

            loading_dataset_task = progress.add_task("🗂️ [cyan]Loading dataset...[/cyan]")
            dataset = iter(import_hf_parquet(dataset_path))
            dataset = chain([next(dataset)], dataset)  # iterator is lazy, force it to actually open the file
            progress.remove_task(loading_dataset_task)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=True,
        ) as progress:
            live.update(progress, refresh=True)
            inference_task = progress.add_task("🔮 [cyan]Running inference...[/cyan]", total=num_tokens_to_generate)

            def progress_callback(event: CollectTracesEvent) -> None:
                progress.update(inference_task, completed=event.tokens_generated)

            traces = inference_collect_traces(
                model,
                dataset,
                num_logits_per_token,
                batch_size,
                max_input_length,
                max_output_length,
                num_tokens_to_generate,
                progress_callback,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb+") as output_fd:
                for trace in traces:
                    blob = trace.serialize()
                    output_fd.write(blob)

            progress.update(inference_task, description="✅ Completed")


@speculator_app.command(help="Train a speculator from inference traces")
def train(
    trace_path: Annotated[
        Path,
        Argument(
            help="File of llm inference traces to train the speculator on",
            metavar="TRACE_PATH",
        ),
    ],
    output_path: Annotated[
        Path,
        Option(
            help="File to save the output to",
            metavar="OUTPUT_PATH",
        ),
    ],
    hashtable_size: Annotated[
        int,
        Option(help="Size of ngram hashtable"),
    ] = 65536,
    num_logits_per_token: Annotated[
        int,
        Option(help="Top K tokens to keep in ngram hashtable"),
    ] = 8,
    ngram_size: Annotated[
        int,
        Option(help="Length of ngrams"),
    ] = 2,
    subsample_size: Annotated[
        int | None,
        Option(
            help="Exit early after training the model on this number of tokens",
            show_default="all",
        ),
    ] = None,
) -> None:
    with open(trace_path, "rb") as trace_fd:
        traces = LalamoCompletion.deserialize_many(trace_fd)

        speculator = NGramSpeculator.new(hashtable_size, num_logits_per_token, ngram_size)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            inference_task = progress.add_task("🔮 [cyan]Training speculator...[/cyan]", total=subsample_size)


            def progress_callback(event: SpeculatorTrainingEvent) -> None:
                progress.update(inference_task, completed=event.trained_tokens)

            train_speculator(speculator, traces, subsample_size, progress_callback)

            progress.update(inference_task, description="✅ Completed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb+") as fd:
        fd.write(speculator.serialize())


@speculator_app.command(help="Run speculator as an autoregressive llm")
def test(
    speculator_path: Annotated[
        Path,
        Argument(
            help="Path to the speculator file.",
            metavar="SPECULATOR_PATH",
        ),
    ],
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory for detokenization.",
            metavar="MODEL_PATH",
        ),
    ],
    seed: Annotated[
        int | None,
        Option(help="Set seed for deterministic sampling"),
    ] = None,
    num_sequences: Annotated[
        int,
        Option(help="Number of sequences to generate"),
    ] = 8,
) -> None:
    model = LanguageModel.load(model_path)

    with open(speculator_path, "rb") as fd:
        speculator = NGramSpeculator.deserialize(fd.read())

    table = Table(
        show_header=False,
        show_lines=True,
        box=box.ROUNDED,
    )

    if seed is not None:
        random.seed(seed)

    for _ in range(num_sequences):
        sequence = test_speculator(speculator)
        detokenized = model.message_processor.detokenize(sequence)
        table.add_row(detokenized)

    console.print(table)


@app.callback()
def _profile_memory(
    ctx: Context,
    profile_memory: Annotated[
        Path | None,
        Option(
            help="Record and save the XLA memory profile to specified path",
            show_default="Don't save the XLA memory profile",
            envvar="LALAMO_PROFILE_MEMORY",
        ),
    ] = None,
) -> None:
    if profile_memory is None:
        return

    if profile_memory.is_dir():
        profile_memory /= "lalamo-memory.prof"

    def _save_memory_profile() -> None:
        console.print(f"Saving XLA memory profile to {profile_memory}")
        jax.profiler.save_device_memory_profile(profile_memory)

    ctx.call_on_close(_save_memory_profile)


if __name__ == "__main__":
    app()
