import re
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from importlib.util import find_spec
from itertools import islice
from pathlib import Path
from typing import Annotated

import jax.profiler
import requests
import soundfile as sf
from click import Context as ClickContext
from click import Parameter as ClickParameter
from click import ParamType
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from typer import Argument, Context, Exit, Option, Typer

from lalamo.audio.utils import play_mono_audio
from lalamo.commands import (
    CollectTracesCallbacks,
    ConversionCallbacks,
    EstimateBatchsizeCallbacks,
    EvalDatasetName,
    GenerateRepliesCallbacks,
    Precision,
    PullCallbacks,
    SpeculatorEvalCallbacks,
    TraceCallbacks,
    TrainCallbacks,
    _suggest_similar_models,
)
from lalamo.commands import collect_traces as _collect_traces
from lalamo.commands import convert as _convert
from lalamo.commands import estimate_batchsize as _estimate_batchsize
from lalamo.commands import generate_replies as _generate_replies
from lalamo.commands import pull as _pull
from lalamo.commands import speculator_eval as _speculator_eval
from lalamo.commands import trace as _trace
from lalamo.common import (
    get_default_device_bytes,
    get_usable_memory_from_bytes,
)
from lalamo.data.lalamo_completions import iter_completions
from lalamo.message_processor import UserMessage
from lalamo.model_import import ModelSpec
from lalamo.model_import.common import FileSpec
from lalamo.model_import.remote_registry import RegistryModel, RegistryModelFile, fetch_available_models
from lalamo.model_registry import ModelRegistry
from lalamo.models import ClassifierModelConfig, LanguageModelConfig
from lalamo.models.common import BatchSizesComputedEvent
from lalamo.models.tts_model import TTSGenerator, TTSMessage
from lalamo.speculator.drafter import Drafter
from lalamo.speculator.drafters import NGramDrafter  # noqa: F401 (triggers Drafter registration)
from lalamo.speculator.eval import EvalQuestion, print_results
from lalamo.speculator.speculate import SamplerConfig, SpeculativeDecodingResult

SCRIPT_NAME = Path(sys.argv[0]).name

DEFAULT_OUTPUT_DIR = Path("models")


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
        repo_to_model = ModelRegistry.build().repo_to_model
        result = repo_to_model.get(value)
        if result is None:
            error_message = f'Model "{value}" not found.'
            error_message += _suggest_similar_models(value, list(repo_to_model))
            return self.fail(error_message, param, ctx)
        return result


class RemoteModelParser(ParamType):
    name: str = "Pre-converted Model"

    def convert(self, value: str, param: ClickParameter | None, ctx: ClickContext | None) -> "RegistryModel":
        try:
            available_models = fetch_available_models()
        except (requests.RequestException, ValueError) as e:
            error_message = f"Failed to fetch model list from SDK. Check your internet connection.\n\nError: {e}"
            return self.fail(error_message, param, ctx)

        repo_to_model = {m.repo_id: m for m in available_models}
        model_spec = repo_to_model.get(value)
        if model_spec is None:
            error_message = f'Model "{value}" not found.'
            error_message += _suggest_similar_models(value, list(repo_to_model))
            return self.fail(error_message, param, ctx)

        return model_spec


def _error(message: str) -> None:
    panel = Panel(message, box=box.ROUNDED, title="Error", title_align="left", border_style="red")
    err_console.print(panel)
    raise Exit(1)


@app.command(help="Chat with a converted model.")
def chat(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory.",
            metavar="MODEL_PATH",
        ),
    ],
    message: Annotated[
        str | None,
        Option(
            help="Message for non-interactive mode",
            show_default="None, run interactively",
        ),
    ] = None,
    max_tokens: Annotated[
        int,
        Option(
            help="Maximum number of tokens to generate per reply.",
        ),
    ] = 8192,
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
        transient=True,
    ) as progress:
        loading_task = progress.add_task("🚀 [cyan]Loading model...[/cyan]")
        model = LanguageModelConfig.load_model(model_path)
        progress.remove_task(loading_task)
        warmup_task = progress.add_task("🔥 Warming up compilation cache...")
        list(model.stream_reply_text([UserMessage("")], max_output_length=1))
        progress.remove_task(warmup_task)

    if message is None:
        console.print(f"🤖 Chatting with [blue]{model_path}[/blue]:")

        messages = []
        while True:
            user_text = console.input("[cyan]user> [/cyan]")
            user_message = UserMessage(user_text)
            messages.append(user_message)

            console.print("[red]assistant> [/red]", end="")
            model_response_tokens = []
            for token in model.stream_reply_text(messages, max_output_length=max_tokens):
                console.print(token, end="")
                model_response_tokens.append(token)
            console.print()
            model_response_text = "".join(model_response_tokens)
            messages.append(model.message_processor.parse_response(model_response_text))
    else:
        for token in model.stream_reply_text([UserMessage(message)], max_output_length=max_tokens):
            console.print(token, end="")
        console.print()


@app.command(help="Classify given message with a Classifier type of model.")
def classify(
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
        model = ClassifierModelConfig.load_model(model_path)
        progress.remove_task(loading_task)
        warmup_task = progress.add_task("🔥 Warming up...")
        model.classify_chat([UserMessage(content="warmup message")])
        progress.remove_task(warmup_task)
    console.print(f"🤖 Classifying input with [blue]{model_path}[/blue]:")
    while True:
        user_text = console.input("[cyan]user> [/cyan]")
        user_message = UserMessage(user_text)

        console.print("[red]assistant> [/red]", end="")
        result = model.classify_chat([user_message])
        for label, confidence in result.items():
            console.print(f"{label} : {confidence}", end="")
        console.print()


@dataclass
class CliConversionCallbacks(ConversionCallbacks):
    overwrite: bool = False

    stack: ExitStack = field(default_factory=ExitStack)
    progress: Progress | None = None
    downloading_tasks: dict[FileSpec, TaskID] = field(default_factory=dict)
    initializing_task: TaskID | None = None
    saving_task: TaskID | None = None

    def started(self) -> None:
        conversion_strs = [
            f"🚀 Converting [cyan]{self.model_spec.name}[/cyan] by [cyan]{self.model_spec.vendor}[/cyan]",
        ]
        if self.precision is not None:
            conversion_strs.append(
                f" and converting floating-point weights into [cyan]{self.precision.name.lower()}[/cyan] precision",
            )
        conversion_strs.append(".")
        console.print("".join(conversion_strs))

        self.progress = self.stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ),
        )

    def output_dir_exists(self) -> None:
        if not self.overwrite and not Confirm().ask(
            rf"⚠️ Output directory [cyan]{self.output_dir}[/cyan] already exists."
            r" Do you want to overwrite it?",
        ):
            raise Exit

        shutil.rmtree(self.output_dir)

    def downloading(self, file_spec: FileSpec) -> None:
        assert self.progress is not None

        self.downloading_tasks[file_spec] = self.progress.add_task(f"Retrieving {file_spec.filename}...")

    def finished_downloading(self, file_spec: FileSpec) -> None:
        assert self.progress is not None

        self.progress.remove_task(self.downloading_tasks[file_spec])

    def initializing_model(self) -> None:
        assert self.progress is not None

        self.initializing_task = self.progress.add_task("Initializing model...")

    def finished_initializing_model(self) -> None:
        assert self.progress is not None
        assert self.initializing_task is not None

        self.progress.remove_task(self.initializing_task)

    def saving_model(self) -> None:
        assert self.progress is not None

        self.saving_task = self.progress.add_task(f"💾 Saving the model to {self.output_dir}")

    def finished_saving_model(self) -> None:
        assert self.progress is not None
        assert self.saving_task is not None

        self.progress.remove_task(self.saving_task)
        self.stack.close()
        console.print(f"🧑‍🍳 Model successfully cooked and saved to [cyan]`{self.output_dir}`[/cyan]!")


@dataclass
class CliPullCallbacks(PullCallbacks):
    stack: ExitStack = field(default_factory=ExitStack)
    progress: Progress | None = None
    downloading_tasks: dict[RegistryModelFile, TaskID] = field(default_factory=dict)

    def started(self) -> None:
        console.print(f"📦 Pulling [cyan]{self.model_spec.name}[/cyan] by [cyan]{self.model_spec.vendor}[/cyan]")

        self.progress = self.stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ),
        )

    def output_dir_exists(self) -> None:
        if not self.overwrite and not Confirm().ask(
            rf"⚠️ Output directory [cyan]{self.output_dir}[/cyan] already exists."
            r" Do you want to overwrite it?",
        ):
            raise Exit

        shutil.rmtree(self.output_dir)

    def downloading(self, file_spec: RegistryModelFile) -> None:
        assert self.progress is not None

        self.downloading_tasks[file_spec] = self.progress.add_task(f"⬇️  Downloading {file_spec.name}...")

    def finished_downloading(self, file_spec: RegistryModelFile) -> None:
        assert self.progress is not None

        self.progress.remove_task(self.downloading_tasks[file_spec])

    def finished(self) -> None:
        assert self.progress is not None

        self.stack.close()
        console.print(f"🎉 Model successfully pulled to [cyan]{self.output_dir}[/cyan]!")


@app.command(help="Synthesize speech from given text utterance")
def tts(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory.",
            metavar="MODEL_PATH",
        ),
    ],
    output_file: Annotated[Path | None, Argument(help="Path to output WAV file with synthesized speech")] = None,
    replay: Annotated[
        bool,
        Option(
            help="Render synthesized speech into default audio interface.",
        ),
    ] = False,
) -> None:
    if output_file is None:
        output_file = Path.cwd() / "generated_speech.wav"
        console.print(f"Will save output to file {output_file}")

    if replay and not find_spec("pyaudio"):
        err_console.print("Failed to import pyaudio package used for audio replay. Run Lalamo without --replay.")
        raise Exit(1)

    console.print(f"🤖 Loading model from specified path: {model_path}.")
    model = TTSGenerator.load_model(model_path)

    assert model is not None
    _stop_word = "/stop"
    while True:
        user_text = console.input(f"[cyan]input text to generate speech({_stop_word} to exit)> [/cyan]")
        if user_text == _stop_word:
            console.print("[green] Goodbye! [/green]")
            break
        if user_text == "":
            continue

        user_message = TTSMessage(content=user_text, speaker_id="speaker:0", style="interleave")

        tts_result = model.generate_speech([user_message])

        if replay:
            play_mono_audio(tts_result.audio, tts_result.audio_params.samplerate)

        if output_file.exists():
            answer = console.input(
                rf"⚠️ Output file [cyan]{output_file}[/cyan] already exists."
                r" Do you want to overwrite it? [cyan]\[y/n][/cyan]: ",
            )
            while answer.lower() not in ["y", "n", "yes", "no"]:
                answer = console.input("Please enter 'y' or 'n': ")
            if answer.lower() in ["y", "yes"]:
                Path.unlink(output_file)
            else:
                console.print("Continue without saving the result")
                continue

        sf.write(str(output_file), tts_result.audio, tts_result.audio_params.samplerate)
        console.print(f"[green] ... saved generated audio to {output_file}[/green]")

        console.print()


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

    _convert(
        model_repo,
        output_dir,
        precision,
        context_length,
        partial(CliConversionCallbacks, overwrite=overwrite),
    )


@app.command(help="Pull a pre-converted model from the SDK repository.")
def pull(
    model_spec: Annotated[
        RegistryModel,
        Argument(
            help=(
                "Model repository ID from the pre-converted catalog. "
                "Example: [cyan]'meta-llama/Llama-3.2-1B-Instruct'[/cyan]. "
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

    _pull(
        model_spec,
        output_dir,
        partial(CliPullCallbacks),
        overwrite=overwrite,
    )


@dataclass
class CliTraceCallbacks(TraceCallbacks):
    overwrite: bool = False

    stack: ExitStack = field(default_factory=ExitStack)
    progress: Progress | None = None
    loading_task: TaskID | None = None
    tracing_task: TaskID | None = None
    saving_task: TaskID | None = None

    def output_exists(self) -> None:
        if not self.overwrite and not Confirm().ask(
            rf"⚠️ Output [cyan]{self.output_path}[/cyan] already exists."
            r" Do you want to overwrite it?",
        ):
            raise Exit

        self.output_path.unlink()

    def started(self) -> None:
        console.print(f"🔍 Tracing [cyan]{self.model_path}[/cyan]")

        self.progress = self.stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ),
        )

    def loading_model(self) -> None:
        assert self.progress is not None

        self.loading_task = self.progress.add_task("🧠 Loading model...")

    def finished_loading_model(self) -> None:
        assert self.progress is not None
        assert self.loading_task is not None

        self.progress.remove_task(self.loading_task)

    def tracing_model(self) -> None:
        assert self.progress is not None

        self.tracing_task = self.progress.add_task("🔍 Recording trace...")

    def finished_tracing_model(self) -> None:
        assert self.progress is not None
        assert self.tracing_task is not None

        self.progress.remove_task(self.tracing_task)

    def saving_trace(self) -> None:
        assert self.progress is not None

        self.saving_task = self.progress.add_task(f"💾 Saving trace to {self.output_path}")

    def finished_saving_trace(self) -> None:
        assert self.progress is not None
        assert self.saving_task is not None

        self.progress.remove_task(self.saving_task)
        self.stack.close()
        console.print(f"💾 Trace saved to [cyan]{self.output_path}[/cyan]")


@app.command(help="Trace a model.")
def trace(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory.",
            metavar="MODEL_PATH",
        ),
    ],
    output_path: Annotated[
        Path | None,
        Option(
            help="Path to save the trace to.",
            show_default="${MODEL_PATH}/traces.safetensors",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing trace file.",
        ),
    ] = False,
    message: Annotated[
        str | None,
        Option(
            help="Text message to use as prompt when recording trace",
        ),
    ] = None,
) -> None:
    if output_path is None:
        output_path = model_path / "traces.safetensors"

    messages = None if message is None else [UserMessage(content=message)]

    _trace(
        model_path,
        output_path,
        messages,
        partial(CliTraceCallbacks, overwrite=overwrite),
    )


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


@dataclass
class CliGenerateRepliesCallbacks(GenerateRepliesCallbacks):
    stack: ExitStack = field(default_factory=ExitStack)
    progress: Progress | None = None
    loading_task: TaskID | None = None
    estimating_task: TaskID | None = None
    generation_task: TaskID | None = None

    def loading_model(self) -> None:
        self.progress = self.stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=True,
            ),
        )
        self.loading_task = self.progress.add_task("🧠 [cyan]Loading model...[/cyan]", total=None)

    def finished_loading_model(self) -> None:
        assert self.progress is not None
        assert self.loading_task is not None
        self.progress.remove_task(self.loading_task)

    def loading_dataset(self) -> None:
        assert self.progress is not None
        self.loading_task = self.progress.add_task("🗂️ [cyan]Loading dataset...[/cyan]", total=None)

    def finished_loading_dataset(self) -> None:
        assert self.progress is not None
        assert self.loading_task is not None
        self.progress.remove_task(self.loading_task)

    def estimating_batchsize(self, sequence_length: int, lo: int, hi: int | None) -> None:
        assert self.progress is not None
        hi_str = str(hi) if hi is not None else "?"
        description = (
            f"📐 [cyan]Computing batch size for the prompt length of {sequence_length}... ({lo}..{hi_str})[/cyan]"
        )
        if self.estimating_task is None:
            self.estimating_task = self.progress.add_task(description)
        else:
            self.progress.update(self.estimating_task, description=description)

    def batch_sizes_estimated(self) -> None:
        assert self.progress is not None
        if self.estimating_task is None:
            self.estimating_task = self.progress.add_task(
                "📐 [cyan]Estimating the best batch sizes...[/cyan]",
                total=None,
            )

    def batch_sizes_computed(self, event: BatchSizesComputedEvent) -> None:
        assert self.progress is not None
        if self.estimating_task is not None:
            self.progress.remove_task(self.estimating_task)
            self.estimating_task = None
        output_console = self.progress.console if self.progress is not None else console
        for info in event.batch_sizes:
            output_console.print(
                f"Prefix length {info.prefix_length} has {info.num_elements} elements, "
                f"with batchsize of {info.batch_size}",
            )
        self.generation_task = self.progress.add_task(
            "🔮 [cyan]Generating replies...[/cyan]",
            total=self.total_rows,
        )

    def generation_progress(self, rows_processed: int) -> None:
        assert self.progress is not None
        assert self.generation_task is not None
        self.progress.update(self.generation_task, completed=rows_processed + 1)

    def finished_generation(self) -> None:
        assert self.progress is not None
        assert self.generation_task is not None
        self.progress.update(self.generation_task, description="✅ Completed")
        self.stack.close()
        console.print(f"💾 Replies saved to [cyan]{self.output_path}[/cyan]")


@app.command(help="Generate replies for conversations in a parquet file.")
def generate_replies(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory.",
            metavar="MODEL_PATH",
        ),
    ],
    dataset_path: Annotated[
        Path,
        Argument(
            help="Path to the input parquet file with conversations.",
            metavar="DATASET_PATH",
        ),
    ],
    output_path: Annotated[
        Path,
        Option(
            help="Path to save the output parquet file.",
        ),
    ],
    vram_gb: Annotated[
        int | None,
        Option(
            help="Maximum VRAM in GB. Batch sizes are estimated automatically.",
            show_default="max on default device",
        ),
    ] = None,
    max_output_length: Annotated[
        int,
        Option(help="Maximum number of tokens to generate per reply."),
    ] = 8192,
    batch_size: Annotated[
        int | None,
        Option(help="Fixed batch size to use, skipping automatic estimation."),
    ] = None,
) -> None:
    if batch_size is not None and vram_gb is not None:
        err_console.print("Cannot use both --batch-size and --vram-gb")
        raise Exit(1)

    max_vram: int | None = None
    if batch_size is None:
        if vram_gb is not None:
            mem_bytes = vram_gb * 1000 * 1000 * 1000
        elif (mem_bytes := get_default_device_bytes()) is None:
            err_console.print("Cannot get the default device's memory stats, use --vram-gb or --batch-size")
            raise Exit(1)

        max_vram = mem_bytes

    _generate_replies(
        model_path=model_path,
        dataset_path=dataset_path,
        output_path=output_path,
        max_vram=max_vram,
        max_output_length=max_output_length,
        batch_size=batch_size,
        callbacks_type=CliGenerateRepliesCallbacks,
    )


speculator_app = Typer()
app.add_typer(speculator_app, name="speculator", help="Train a speculator for a model.")


@dataclass
class CliEstimateBatchsizeCallbacks(EstimateBatchsizeCallbacks):
    stack: ExitStack = field(default_factory=ExitStack)
    loading_task: TaskID | None = None
    estimating_task: TaskID | None = None

    def loading_model(self) -> None:
        self.progress = self.stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ),
        )
        self.loading_task = self.progress.add_task("[cyan]Loading model...[/cyan]")

    def finished_loading_model(self) -> None:
        assert self.loading_task is not None
        self.progress.remove_task(self.loading_task)

    def estimating_batchsize(self, lo: int, hi: int | None) -> None:
        hi_str = str(hi) if hi is not None else "?"
        description = f"[cyan]Estimating batch size... ({lo}..{hi_str})[/cyan]"
        if self.estimating_task is None:
            self.estimating_task = self.progress.add_task(description)
        else:
            self.progress.update(self.estimating_task, description=description)

    def finished_estimating_batchsize(self, batchsize: int) -> None:
        if self.estimating_task is not None:
            self.progress.remove_task(self.estimating_task)
        self.stack.close()
        console.print(f"Found maximum batch size: [cyan]{batchsize}[/cyan]")


@speculator_app.command(help="Estimate maximum batch size at which a model can be run.")
def estimate_batchsize(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory",
            metavar="MODEL_PATH",
        ),
    ],
    max_input_length: Annotated[
        int,
        Option(help="Max input length of a model."),
    ] = 1024,
    max_output_length: Annotated[
        int,
        Option(help="Max output length of a model."),
    ] = 1024,
    num_logits_per_token: Annotated[
        int,
        Option(help="Number of top logits that will be recorded."),
    ] = 1024,
    vram_gb: Annotated[
        int | None,
        Option(
            help="Maximum vram size in gb allowed.",
            show_default="max on default device",
        ),
    ] = None,
) -> None:
    if vram_gb is not None:
        # note that in practice GPUs use GiB in their docs, e.g. H100 actually has 85GB of memory
        mem_bytes = vram_gb * 1000 * 1000 * 1000
    elif (mem_bytes := get_default_device_bytes()) is None:
        err_console.print("Cannot get the default device's memory stats, use --vram-gb")
        raise Exit(1)

    usable_mem = get_usable_memory_from_bytes(mem_bytes)

    callbacks_type = CliEstimateBatchsizeCallbacks

    _estimate_batchsize(
        model_path,
        usable_mem,
        max_input_length,
        max_output_length,
        num_logits_per_token,
        callbacks_type,
    )


@dataclass
class CliCollectTracesCallbacks(CollectTracesCallbacks):
    stack: ExitStack = field(default_factory=ExitStack)
    live: Live | None = None
    loading_task: TaskID | None = None
    inference_task: TaskID | None = None

    def loading_model(self) -> None:
        self.live = self.stack.enter_context(Live(refresh_per_second=10))
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        self.live.update(self.progress, refresh=True)
        self.loading_task = self.progress.add_task("🧠 [cyan]Loading model...[/cyan]")

    def finished_loading_model(self) -> None:
        assert self.loading_task is not None
        self.progress.remove_task(self.loading_task)

    def loading_dataset(self) -> None:
        self.loading_task = self.progress.add_task("🗂️ [cyan]Loading dataset...[/cyan]")

    def finished_loading_dataset(self) -> None:
        assert self.loading_task is not None
        assert self.live is not None
        self.progress.remove_task(self.loading_task)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.live.update(self.progress, refresh=True)
        self.inference_task = self.progress.add_task(
            "🔮 [cyan]Running inference...[/cyan]",
            total=self.num_tokens_to_generate,
        )

    def inference_progress(self, tokens_generated: int) -> None:
        assert self.inference_task is not None
        self.progress.update(self.inference_task, completed=tokens_generated)

    def finished_inference(self) -> None:
        assert self.inference_task is not None
        self.progress.update(self.inference_task, description="✅ Completed")
        self.stack.close()


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
            help="Directory to save sharded trace files to",
            metavar="OUTPUT_PATH",
        ),
    ],
    num_logits_per_token: Annotated[
        int,
        Option(help="Record logits for this number of most probable tokens"),
    ] = 1024,
    trace_layers: Annotated[
        list[int] | None,
        Option(help="0-based transformer layer indices to save hidden states for"),
    ] = None,
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
    shard_size: Annotated[
        int,
        Option(help="Number of completions to store in each output shard"),
    ] = 64,
    num_tokens_to_generate: Annotated[
        int | None,
        Option(
            help="Exit early after generating this number of output tokens",
            show_default="all",
        ),
    ] = None,
) -> None:
    _collect_traces(
        model_path,
        dataset_path,
        output_path,
        num_logits_per_token,
        tuple(trace_layers or []),
        max_input_length,
        max_output_length,
        batch_size,
        shard_size,
        num_tokens_to_generate,
        CliCollectTracesCallbacks,
    )


@speculator_app.command(help="View model inference traces")
def view_traces(
    trace_path: Annotated[
        Path,
        Argument(
            help="Trace directory to view.",
            metavar="TRACE_PATH",
        ),
    ],
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory for detokenization.",
            metavar="MODEL_PATH",
        ),
    ],
    num_completions: Annotated[
        int | None,
        Option(
            help="Number of completions to show.",
        ),
    ] = None,
) -> None:
    model = LanguageModelConfig.load_model(model_path)
    traces = iter_completions(trace_path)

    table = Table(
        show_lines=True,
        box=box.ROUNDED,
    )
    table.add_column("Prefix")
    table.add_column("Completion")

    from rich.text import Text

    for completion in islice(traces, num_completions):
        detokenized_prefix = model.message_processor.detokenize(completion.prefix_token_ids)
        detokenized_completion = model.message_processor.detokenize(completion.completion_token_ids)
        table.add_row(Text(detokenized_prefix), Text(detokenized_completion))

    console.print(table)


@dataclass
class CliTrainCallbacks(TrainCallbacks):
    stack: ExitStack = field(default_factory=ExitStack)
    training_task: TaskID | None = None

    def training_progress(self, trained_tokens: int) -> None:
        if self.training_task is None:
            self.progress = self.stack.enter_context(
                Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                ),
            )
            self.training_task = self.progress.add_task(
                "🔮 [cyan]Training speculator...[/cyan]",
                total=self.subsample_size,
            )
        self.progress.update(self.training_task, completed=trained_tokens)

    def finished_training(self) -> None:
        if self.training_task is not None:
            self.progress.update(self.training_task, description="✅ Completed")
            self.progress.remove_task(self.training_task)
            self.stack.close()

    def finished_saving(self) -> None:
        console.print(f"💾 Speculator saved to [cyan]{self.output_path}[/cyan]")


for drafter_cls in Drafter.registered_types():
    drafter_cls.train_command(speculator_app, CliTrainCallbacks)


@dataclass
class CliSpeculatorEvalCallbacks(SpeculatorEvalCallbacks):
    stack: ExitStack = field(default_factory=ExitStack)
    live: Live | None = None
    progress: Progress | None = None
    setup_task: TaskID | None = None
    eval_task: TaskID | None = None
    running_accepted: int = 0
    running_steps: int = 0
    running_tokens: int = 0
    running_elapsed_s: float = 0.0

    def _ensure_live(self) -> None:
        if self.live is None:
            self.live = self.stack.enter_context(Live(refresh_per_second=10))
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            )
            self.live.update(self.progress, refresh=True)

    def loading_model(self) -> None:
        self._ensure_live()
        assert self.progress is not None
        self.setup_task = self.progress.add_task("🧠 [cyan]Loading model...[/cyan]")

    def finished_loading_model(self) -> None:
        assert self.progress is not None
        assert self.setup_task is not None
        self.progress.remove_task(self.setup_task)

    def loading_drafter(self) -> None:
        assert self.progress is not None
        self.setup_task = self.progress.add_task("🔮 [cyan]Loading drafter...[/cyan]")

    def finished_loading_drafter(self) -> None:
        assert self.progress is not None
        assert self.setup_task is not None
        self.progress.remove_task(self.setup_task)

    def loading_dataset(self) -> None:
        assert self.progress is not None
        self.setup_task = self.progress.add_task(
            f"🗂️ [cyan]Loading dataset ({self.dataset_name.value})...[/cyan]",
        )

    def finished_loading_dataset(self, num_questions: int) -> None:
        assert self.progress is not None
        assert self.setup_task is not None
        self.progress.update(
            self.setup_task,
            description=f"🗂️ [green]Loaded {num_questions} questions[/green]",
        )
        self.progress.remove_task(self.setup_task)

    def eval_started(self, num_questions: int) -> None:
        assert self.live is not None
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.live.update(self.progress, refresh=True)
        self.eval_task = self.progress.add_task(
            f"⚡ [cyan]Evaluating {self.dataset_name.value}[/cyan] (draft_acc=—)",
            total=num_questions,
        )

    def eval_progress(
        self,
        question_idx: int,
        question: EvalQuestion,
        result: SpeculativeDecodingResult,
        elapsed_s: float,
    ) -> None:
        assert self.progress is not None
        assert self.eval_task is not None
        self.running_accepted += result.total_accepted
        self.running_steps += result.num_steps
        self.running_tokens += len(result.generated)
        self.running_elapsed_s += elapsed_s
        running_acc = self.running_accepted / max(self.running_steps, 1)
        running_tps = self.running_tokens / self.running_elapsed_s if self.running_elapsed_s > 0 else 0.0
        self.progress.update(
            self.eval_task,
            advance=1,
            description=(
                f"⚡ [cyan]Evaluating {self.dataset_name.value}[/cyan] "
                f"#{question_idx + 1} [{question.category}] "
                f"draft_acc={running_acc:.2f} tok/s={running_tps:.1f}"
            ),
        )

    def finished_eval(self) -> None:
        assert self.progress is not None
        assert self.eval_task is not None
        self.progress.update(self.eval_task, description="✅ Completed")
        self.stack.close()


@speculator_app.command(name="eval", help="Evaluate speculative decoding on a benchmark")
def speculate_eval(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the lalamo model directory",
            metavar="MODEL_PATH",
        ),
    ],
    speculator_path: Annotated[
        Path,
        Argument(
            help="Path to the trained speculator file",
            metavar="SPECULATOR_PATH",
        ),
    ],
    dataset: Annotated[
        EvalDatasetName,
        Option(help="Benchmark to evaluate on"),
    ] = EvalDatasetName.MTBENCH,
    width: Annotated[
        int,
        Option(help="Max children per trie node"),
    ] = 4,
    depth: Annotated[
        int,
        Option(help="Max speculation depth (K)"),
    ] = 8,
    max_tokens: Annotated[
        int,
        Option(help="Max tokens per question"),
    ] = 2048,
    num_questions: Annotated[
        int | None,
        Option(
            help="Limit the number of questions evaluated",
            show_default="full dataset",
        ),
    ] = None,
    drafter_name: Annotated[
        str,
        Option(help="Drafter type (ngram, medusa)"),
    ] = "ngram",
    mtbench_cache_path: Annotated[
        Path,
        Option(help="Local cache path for MT-Bench questions (used only for --dataset mtbench)"),
    ] = Path("mtbench_questions.jsonl"),
) -> None:
    results = _speculator_eval(
        model_path=model_path,
        speculator_path=speculator_path,
        dataset_name=dataset,
        num_questions=num_questions,
        mtbench_cache_path=mtbench_cache_path,
        sampler_config=SamplerConfig(width=width, K=depth, max_tokens=max_tokens),
        drafter_name=drafter_name,
        callbacks_type=CliSpeculatorEvalCallbacks,
    )
    print_results(results)


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
