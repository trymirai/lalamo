import random
import re
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Annotated

import jax.profiler
import thefuzz.process
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

from lalamo.commands import (
    CollectTracesCallbacks,
    ConversionCallbacks,
    EstimateBatchsizeCallbacks,
    Precision,
    TrainCallbacks,
)
from lalamo.commands import collect_traces as _collect_traces
from lalamo.commands import convert as _convert
from lalamo.commands import estimate_batchsize as _estimate_batchsize
from lalamo.commands import train as _train
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.message_processor import UserMessage
from lalamo.model_import import REPO_TO_MODEL, ModelSpec
from lalamo.model_import.common import FileSpec
from lalamo.models import ClassifierModelConfig, LanguageModelConfig
from lalamo.speculator.estimator import get_default_device_memory
from lalamo.speculator.ngram import NGramSpeculator
from lalamo.speculator.utils import test_speculator

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
        model = LanguageModelConfig.load_model(model_path)
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
        self.downloading_tasks[file_spec] = self.progress.add_task(f"Retrieving {file_spec.filename}...")

    def finished_downloading(self, file_spec: FileSpec) -> None:
        self.progress.remove_task(self.downloading_tasks[file_spec])

    def initializing_model(self) -> None:
        self.initializing_task = self.progress.add_task("Initializing model...")

    def finished_initializing_model(self) -> None:
        assert self.initializing_task is not None

        self.progress.remove_task(self.initializing_task)

    def saving_model(self) -> None:
        self.saving_task = self.progress.add_task(f"💾 Saving the model to {self.output_dir}")

    def finished_saving_model(self) -> None:
        assert self.saving_task is not None

        self.progress.remove_task(self.saving_task)
        self.stack.close()
        console.print(f"🧑‍🍳 Model successfully cooked and saved to [cyan]`{self.output_dir}`[/cyan]!")


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
    message_for_trace: Annotated[
        str | None,
        Option(
            help="Text message to use as prompt when recording trace",
        ),
    ] = None,
) -> None:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / model_repo.name

    _convert(
        model_repo,
        output_dir,
        precision,
        context_length,
        include_traces,
        message_for_trace,
        partial(CliConversionCallbacks, overwrite=overwrite),
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
    ] = 8,
    vram_gb: Annotated[
        int | None,
        Option(
            help="Maximum vram size in gb allowed.",
            show_default="max on default device",
        ),
    ] = None,
) -> None:
    if vram_gb is not None:
        mem = vram_gb * 1024 * 1024 * 1024
    elif (mem := get_default_device_memory()) is None:
        err_console.print("Cannot get the default device's memory stats, use --vram-gb")
        raise Exit(1)

    callbacks_type = CliEstimateBatchsizeCallbacks

    _estimate_batchsize(model_path, mem, max_input_length, max_output_length, num_logits_per_token, callbacks_type)


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
    _collect_traces(
        model_path,
        dataset_path,
        output_path,
        num_logits_per_token,
        max_input_length,
        max_output_length,
        batch_size,
        num_tokens_to_generate,
        CliCollectTracesCallbacks,
    )


@speculator_app.command(help="View model inference traces")
def view_traces(
    trace_path: Annotated[
        Path,
        Argument(
            help="File of inference traces to view.",
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

    with open(trace_path, "rb") as trace_fd:
        traces = LalamoCompletion.deserialize_many(trace_fd)

        table = Table(
            show_lines=True,
            box=box.ROUNDED,
        )
        table.add_column("Prefix")
        table.add_column("Completion")

        for completion in islice(traces, num_completions):
            detokenized_prefix = model.message_processor.detokenize(completion.prefix_token_ids)
            detokenized_completion = model.message_processor.detokenize(completion.completion_token_ids)
            table.add_row(detokenized_prefix, detokenized_completion)

        console.print(table)


@dataclass
class CliTrainCallbacks(TrainCallbacks):
    stack: ExitStack = field(default_factory=ExitStack)
    training_task: TaskID | None = None

    def started(self) -> None:
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

    def training_progress(self, trained_tokens: int) -> None:
        assert self.training_task is not None
        self.progress.update(self.training_task, completed=trained_tokens)

    def finished_training(self) -> None:
        assert self.training_task is not None
        self.progress.update(self.training_task, description="✅ Completed")
        self.progress.remove_task(self.training_task)
        self.stack.close()

    def saving_speculator(self) -> None:
        pass

    def finished_saving_speculator(self) -> None:
        console.print(f"💾 Speculator saved to [cyan]{self.output_path}[/cyan]")


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
    _train(
        trace_path,
        output_path,
        hashtable_size,
        num_logits_per_token,
        ngram_size,
        subsample_size,
        CliTrainCallbacks,
    )


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
    model = LanguageModelConfig.load_model(model_path)

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
