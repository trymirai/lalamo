import re
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated

import jax
import jax.profiler
import requests
import soundfile as sf
from click import Context as ClickContext
from click import Parameter as ClickParameter
from click import ParamType
from rich import box
from rich.console import Console, Group
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
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from typer import Argument, Exit, Option, Typer

from lalamo.audio.utils import play_mono_audio
from lalamo.commands import (
    ConversionCallbacks,
    DType,
    EvalDatasetName,
    EvalResults,
    PullCallbacks,
    _suggest_similar_models,
)
from lalamo.commands import convert as _convert
from lalamo.commands import evaluate_speculator as _evaluate_speculator
from lalamo.commands import pull as _pull
from lalamo.model_import import ModelSpec
from lalamo.model_import.common import FileSpec
from lalamo.model_import.remote_registry import RegistryModel, RegistryModelFile, fetch_available_models
from lalamo.model_registry import ModelRegistry
from lalamo.models import GenerationConfig, LanguageModel, TTSModel
from lalamo.models.chat_codec import Message, UserMessage
from lalamo.models.tts_codec import TTSMessage
from lalamo.module import Keychain
from lalamo.speculator.common import load_speculator
from lalamo.utils.memory import get_available_bytes_on_default_device

SCRIPT_NAME = Path(sys.argv[0]).name

DEFAULT_OUTPUT_DIR = Path("models")


console = Console()
err_console = Console(stderr=True)
app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
speculator_app = Typer(no_args_is_help=True)
app.add_typer(speculator_app, name="speculator", help="Speculator utilities.")


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
            help="Message for non-interactive mode.",
            show_default="None, run interactively",
        ),
    ] = None,
    max_tokens: Annotated[
        int,
        Option(
            help="Maximum number of tokens to generate per reply.",
        ),
    ] = 8192,
    speculator_path: Annotated[
        Path | None,
        Option(
            "--speculator",
            help="Path to a speculator artifact.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            show_default="No speculator",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        Option(
            help="Sampling temperature. Use 0 for greedy decoding.",
            show_default="model default",
        ),
    ] = None,
) -> None:
    generation_config: GenerationConfig | None = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
        transient=True,
    ) as progress:
        loading_task = progress.add_task("🚀 [cyan]Loading model...[/cyan]")
        model = LanguageModel.load(model_path)
        speculator = None if speculator_path is None else load_speculator(speculator_path, model.decoder)
        if temperature is not None:
            generation_config = replace(model.config.generation_config, temperature=temperature)
        progress.remove_task(loading_task)
        warmup_task = progress.add_task("🔥 Warming up compilation cache...")
        warmup_tokens = iter(
            model.stream_reply_text(
                [UserMessage("")],
                generation_config=generation_config,
                max_output_length=max_tokens,
                speculator=speculator,
                keychain=Keychain.init(0),
            ),
        )
        for _ in range(2):
            try:
                next(warmup_tokens)
            except StopIteration:
                break
        progress.remove_task(warmup_task)

    def print_response(response_messages: list[Message], keychain: Keychain) -> str:
        response_text_parts = []
        steps = 0
        tokens = 0
        started_at = time.perf_counter()

        def status_text() -> Text:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            return Text.assemble(
                ("tok/step: ", "dim"),
                (f"{tokens / max(steps, 1):.2f}", "cyan"),
                (" | throughput: ", "dim"),
                (f"{tokens / elapsed:.2f}", "green"),
                (" tokens/sec", "dim"),
            )

        def render_response() -> Group:
            return Group(
                Text("".join(response_text_parts)),
                Rule(style="dim"),
                status_text(),
            )

        with Live(
            render_response(),
            console=console,
            refresh_per_second=8,
            transient=False,
            vertical_overflow="visible",
        ) as live:
            def step_callback(num_tokens: int) -> None:
                nonlocal steps, tokens
                steps += 1
                tokens += num_tokens
                live.update(render_response())

            for token in model.stream_reply_text(
                response_messages,
                generation_config=generation_config,
                max_output_length=max_tokens,
                speculator=speculator,
                step_callback=step_callback,
                keychain=keychain,
            ):
                response_text_parts.append(token)
                live.update(render_response())
        return "".join(response_text_parts)

    if message is None:
        console.print(f"🤖 Chatting with [blue]{model_path}[/blue]:")
        messages: list[Message] = []
        turn_index = 0
        while True:
            user_text = console.input("[cyan]user> [/cyan]")
            messages.append(UserMessage(user_text))

            console.print("[red]assistant> [/red]")
            response_text = print_response(messages, Keychain.init(turn_index + 1))
            messages.append(model.token_codec.parse_response(response_text))
            turn_index += 1
    else:
        print_response([UserMessage(message)], Keychain.init(1))


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
        if self.dtype is not None:
            conversion_strs.append(
                f" and loading floating-point weights as [cyan]{self.dtype.name.lower()}[/cyan]",
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


@app.command(help="Synthesize speech from given text utterance.")
def tts(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory.",
            metavar="MODEL_PATH",
        ),
    ],
    output_file: Annotated[Path | None, Argument(help="Path to output WAV file with synthesized speech.")] = None,
    replay: Annotated[
        bool,
        Option(
            help="Render synthesized speech into default audio interface.",
        ),
    ] = False,
    message: Annotated[
        str | None,
        Option(
            help="Message for non-interactive mode.",
            show_default="None, run interactively",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing output file without prompting. Always enabled with --message.",
        ),
    ] = False,
) -> None:
    if output_file is None:
        output_file = Path.cwd() / "generated_speech.wav"
        console.print(f"Will save output to file {output_file}")

    if replay and find_spec("pyaudio") is None:
        err_console.print("Failed to import pyaudio package used for audio replay. Run Lalamo without --replay.")
        raise Exit(1)

    console.print(f"🤖 Loading model from specified path: {model_path}.")
    model = TTSModel.load(model_path)

    keychain = Keychain.init(0)
    messages = [message] if message is not None else None
    overwrite_existing_output = overwrite or message is not None
    while True:
        if messages is not None:
            try:
                user_text = messages.pop(0)
            except IndexError:
                break
        else:
            user_text = console.input("[cyan]input text to generate speech> [/cyan]")
        if user_text == "":
            continue

        user_message = TTSMessage(content=user_text, speaker_id="speaker:0", style="interleave")
        keychain, generation_keychain = keychain.split()
        tts_result = model.generate_speech([user_message], keychain=generation_keychain)

        if replay:
            play_mono_audio(tts_result.audio, tts_result.audio_params.samplerate)

        if output_file.exists():
            if overwrite_existing_output:
                output_file.unlink()
            else:
                answer = console.input(
                    rf"⚠️ Output file [cyan]{output_file}[/cyan] already exists."
                    r" Do you want to overwrite it? [cyan]\[y/n][/cyan]: ",
                )
                while answer.lower() not in ["y", "n", "yes", "no"]:
                    answer = console.input("Please enter 'y' or 'n': ")
                if answer.lower() in ["y", "yes"]:
                    output_file.unlink()
                else:
                    console.print("Continue without saving the result")
                    continue

        sf.write(str(output_file), tts_result.audio, tts_result.audio_params.samplerate)
        console.print(f"[green] ... saved generated audio to {output_file}[/green]")
        console.print()


@app.command(help="Import and export a model into the local Lalamo format.")
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
    dtype: Annotated[
        DType | None,
        Option(
            help="Dtype to use for activations and non-quantized weights.",
            show_default="Native dtype of the model",
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
        dtype,
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
            console.print(spec.origin.description)
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
    table.add_column("Origin", justify="left", style="cyan", no_wrap=True)
    for spec in sorted_specs:
        table.add_row(
            spec.vendor,
            spec.family,
            spec.size,
            spec.origin.description,
        )
    console.print(table)


@app.command(help="Start a server for batched inference.")
def server(
    host: Annotated[
        str,
        Option(help="Host to bind to."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        Option(help="Port to bind to."),
    ] = 8293,
    vram_gb: Annotated[
        float | None,
        Option(
            help="Maximum VRAM in GB. Batch sizes are estimated automatically.",
            show_default="max on default device",
        ),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        Option(
            help="Directory to persist completed batches to.",
            show_default="~/.cache/lalamo/batches",
        ),
    ] = None,
) -> None:
    try:
        from lalamo.server import start_server  # noqa: PLC0415
    except ImportError as error:
        err_console.print("Server extras not installed. Install with: uv add 'lalamo[server]'")
        raise Exit(1) from error

    if vram_gb is not None:
        vram_bytes = int(vram_gb * 1000 * 1000 * 1000)
    elif (vram_bytes := get_available_bytes_on_default_device()) is None:
        err_console.print("Cannot get the default device's memory stats, use --vram-gb")
        raise Exit(1)

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "lalamo" / "batches"

    start_server(host=host, port=port, vram_bytes=vram_bytes, cache_dir=cache_dir)


def print_speculator_eval_results(results: EvalResults) -> None:
    config = results.config
    speculator_name = config.speculator_path.name if config.speculator_path is not None else "no-speculator"
    dataset_label = ",".join(name.value for name in config.dataset_names)
    label = f"{dataset_label}, {speculator_name}"
    config_table = Table(
        title=f"Speculator eval config ({label})",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    config_table.add_column("Key")
    config_table.add_column("Value")
    config_table.add_row("dataset", dataset_label)
    config_table.add_row("model_path", str(config.model_path))
    config_table.add_row("speculator", str(config.speculator_path) if config.speculator_path is not None else "none")
    config_table.add_row("questions", str(config.num_questions))
    config_table.add_row("batch_size", str(config.batch_size))
    config_table.add_row("max_output_length", str(config.max_output_length))
    config_table.add_row("reasoning", str(config.reasoning).lower())
    config_table.add_row("temperature", str(config.temperature))
    config_table.add_row("top_p", "none" if config.top_p is None else str(config.top_p))
    config_table.add_row("top_k", "none" if config.top_k is None else str(config.top_k))
    config_table.add_row("min_p", "none" if config.min_p is None else str(config.min_p))
    config_table.add_row("padded_length", str(config.padded_length))
    config_table.add_row("warmup", str(config.warmup).lower())
    config_table.add_row("seed", str(config.seed))
    config_table.add_row("mtbench_cache", str(config.mtbench_cache_path))
    console.print(config_table)

    table = Table(
        title=f"Speculator evaluation ({label})",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    table.add_column("Category", justify="right")
    table.add_column("tok/step", justify="right")
    table.add_column("tok/sec", justify="right")
    table.add_column("mal", justify="right")
    table.add_column("spec_rate", justify="right")
    table.add_column("questions", justify="right")

    for category in sorted(results.by_category):
        stats = results.by_category[category]
        table.add_row(
            category,
            f"{stats.tokens_per_step:.2f}",
            f"{stats.tokens_per_second:.2f}",
            f"{stats.mean_draft_accepted:.2f}",
            f"{stats.speculation_rate:.2%}",
            str(stats.count),
        )
    table.add_section()
    table.add_row(
        "OVERALL",
        f"{results.tokens_per_step:.2f}",
        f"{results.tokens_per_second:.2f}",
        f"{results.mean_draft_accepted:.2f}",
        f"{results.speculation_rate:.2%}",
        str(results.total_count),
    )
    console.print(table)


def parse_eval_dataset_names(value: str) -> tuple[EvalDatasetName, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("--dataset must specify at least one dataset.")
    available = {dataset.value: dataset for dataset in EvalDatasetName}
    unknown = tuple(name for name in names if name not in available)
    if unknown:
        raise ValueError(f"Unknown eval dataset(s): {', '.join(unknown)}. Available: {', '.join(sorted(available))}.")
    return tuple(available[name] for name in names)


@speculator_app.command("eval", help="Evaluate speculative decoding MAL and throughput.")
def eval_speculator(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the model directory.",
            metavar="MODEL_PATH",
        ),
    ],
    speculator_path: Annotated[
        Path | None,
        Option(
            "--speculator",
            help="Path to a speculator artifact file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            show_default="none",
        ),
    ] = None,
    dataset_name: Annotated[
        str,
        Option("--dataset", help="Comma-separated evaluation datasets: mtbench,gsm8k,humaneval,math500."),
    ] = "gsm8k,mtbench,math500",
    num_questions: Annotated[
        int | None,
        Option("--num_questions", "--num-questions", help="Number of questions to evaluate."),
    ] = None,
    batch_size: Annotated[
        int,
        Option("--batch_size", "--batch-size", help="Batch size used for generation."),
    ] = 32,
    max_output_length: Annotated[
        int,
        Option("--max_output_length", "--max-output-length", help="Maximum number of generated tokens per question."),
    ] = 4096,
    reasoning: Annotated[
        bool,
        Option("--reasoning/--no-reasoning", help="Render eval prompts with reasoning/thinking enabled."),
    ] = True,
    temperature: Annotated[
        float,
        Option("--temperature", help="Sampling temperature. Use 0 for greedy decoding."),
    ] = 1.0,
    top_p: Annotated[
        float | None,
        Option("--top_p", "--top-p", help="Nucleus sampling threshold.", show_default="none"),
    ] = None,
    top_k: Annotated[
        int | None,
        Option("--top_k", "--top-k", help="Top-k sampling cutoff.", show_default="none"),
    ] = None,
    min_p: Annotated[
        float | None,
        Option("--min_p", "--min-p", help="Min-p sampling cutoff.", show_default="none"),
    ] = None,
    mtbench_cache_path: Annotated[
        Path | None,
        Option(
            "--mtbench_cache",
            "--mtbench-cache",
            help="Cache path for MT-Bench questions.",
            show_default="~/.cache/lalamo/eval/mt_bench_questions.jsonl",
        ),
    ] = None,
    seed: Annotated[
        int,
        Option("--seed", help="Sampling seed."),
    ] = 0,
    warmup: Annotated[
        bool,
        Option("--warmup/--no-warmup", help="Run one warmup generation before measuring throughput."),
    ] = True,
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        transient=True,
    ) as progress:
        progress_task = progress.add_task("📊 Evaluating speculative decoding...", total=None)
        cache_path = mtbench_cache_path or Path.home() / ".cache" / "lalamo" / "eval" / "mt_bench_questions.jsonl"
        results = _evaluate_speculator(
            model_path=model_path,
            dataset_names=parse_eval_dataset_names(dataset_name),
            speculator_path=speculator_path,
            mtbench_cache_path=cache_path,
            num_questions=num_questions,
            batch_size=batch_size,
            max_output_length=max_output_length,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            warmup=warmup,
            progress_callback=lambda completed, total: progress.update(
                progress_task,
                completed=completed,
                total=total,
            ),
        )
    print_speculator_eval_results(results)


@app.callback()
def _profile_memory(
    ctx: ClickContext,
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
