import re
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import Annotated

import jax
import jax.numpy as jnp
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
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text
from typer import Argument, Exit, Option, Typer

from lalamo.audio.utils import play_mono_audio
from lalamo.commands import (
    ConversionCallbacks,
    DType,
    PullCallbacks,
    SpeculatorConversionCallbacks,
    _suggest_similar_models,
)
from lalamo.commands import convert as _convert
from lalamo.commands import convert_speculator as _convert_speculator
from lalamo.commands import pull as _pull
from lalamo.model_import import ModelSpec
from lalamo.model_import.common import FileSpec
from lalamo.model_import.remote_registry import RegistryModel, RegistryModelFile, fetch_available_models
from lalamo.model_registry import ModelRegistry
from lalamo.models import ClassifierModel, GenerationConfig, LanguageModel, TTSModel
from lalamo.models.chat_codec import Message, UserMessage
from lalamo.models.tts_codec import TTSMessage
from lalamo.module import Keychain
from lalamo.speculator import NoSpeculator, Speculator, import_speculator
from lalamo.utils.memory import get_available_bytes_on_default_device
from lalamo.utils.sharding import ShardingConfig

SCRIPT_NAME = Path(sys.argv[0]).name

DEFAULT_OUTPUT_DIR = Path("models")


console = Console()
err_console = Console(stderr=True)
app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
speculator_app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
app.add_typer(speculator_app, name="speculator", help="Manage speculator artifacts.")


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

        model_by_artifact_repo = {model.artifact_repo_id: model for model in available_models}
        model_spec = model_by_artifact_repo.get(value)
        if model_spec is not None:
            return model_spec

        identifiers = sorted(model_by_artifact_repo)
        error_message = f'Model "{value}" not found.'
        error_message += _suggest_similar_models(value, identifiers)
        return self.fail(error_message, param, ctx)


def _error(message: str) -> None:
    panel = Panel(message, box=box.ROUNDED, title="Error", title_align="left", border_style="red")
    err_console.print(panel)
    raise Exit(1)


def _stream_reply_with_stats(
    model: LanguageModel,
    messages: list[Message],
    generation_config: GenerationConfig | None,
    max_tokens: int,
    keychain: Keychain,
    speculator: Speculator,
    *,
    enable_thinking: bool,
) -> str:
    prompt_token_ids = jnp.asarray(
        model.token_codec.encode_request(messages, enable_thinking=enable_thinking),
        dtype=jnp.int32,
    )
    response_token_ids: list[int] = []
    response_text = ""
    num_steps = 0
    start_time = time.perf_counter()

    def render() -> Group:
        elapsed_seconds = time.perf_counter() - start_time
        num_tokens = len(response_token_ids)
        throughput = num_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
        tokens_per_step = num_tokens / num_steps if num_steps > 0 else 0.0
        return Group(
            Text.assemble(("assistant> ", "red"), response_text),
            Text(
                f"⚡ {throughput:.1f} tok/s · {tokens_per_step:.2f} tok/step · {num_tokens} tokens",
                style="dim",
            ),
        )

    with Live(render(), console=console, refresh_per_second=8, vertical_overflow="visible") as live:
        for block_token_ids in model.stream_token_blocks(
            prompt_token_ids,
            generation_config=generation_config,
            max_output_length=max_tokens,
            keychain=keychain,
            speculator=speculator,
        ):
            num_steps += 1
            response_token_ids.extend(int(token_id.item()) for token_id in block_token_ids)
            response_text = model.token_codec.decode_tokens(response_token_ids, hide_invalid_utf_chars=True)
            live.update(render())
    return response_text


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
    temperature: Annotated[
        float | None,
        Option(
            help="Sampling temperature. Use 0 for greedy decoding.",
            show_default="model default",
        ),
    ] = None,
    speculator_path: Annotated[
        Path | None,
        Option(
            "--speculator",
            help="Path to a converted speculator directory for speculative decoding.",
            show_default="None, decode without speculation",
        ),
    ] = None,
    no_thinking: Annotated[
        bool,
        Option(
            "--no-thinking",
            help="Disable thinking mode in the chat template.",
        ),
    ] = False,
) -> None:
    generation_config: GenerationConfig | None = None
    speculator: Speculator = NoSpeculator()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
        transient=True,
    ) as progress:
        loading_task = progress.add_task("🚀 [cyan]Loading model...[/cyan]")
        model = LanguageModel.load(model_path, ShardingConfig.replicated())
        if speculator_path is not None:
            speculator = import_speculator(speculator_path, sharding_config=ShardingConfig.replicated())
        if temperature is not None:
            generation_config = replace(model.config.generation_config, temperature=temperature)
        progress.remove_task(loading_task)

    with jax.set_mesh(model.sharding_config.mesh):
        if message is None:
            console.print(f"🤖 Chatting with [blue]{model_path}[/blue]:")
            messages: list[Message] = []
            turn_index = 0
            while True:
                user_text = console.input("[cyan]user> [/cyan]")
                messages.append(UserMessage(user_text))

                response_text = _stream_reply_with_stats(
                    model,
                    messages,
                    generation_config,
                    max_tokens,
                    Keychain.init(turn_index + 1, sharding_config=model.sharding_config),
                    speculator,
                    enable_thinking=not no_thinking,
                )
                messages.append(model.token_codec.parse_response(response_text, expect_thinking=not no_thinking))
                turn_index += 1

        else:
            _stream_reply_with_stats(
                model,
                [UserMessage(message)],
                generation_config,
                max_tokens,
                Keychain.init(1, sharding_config=model.sharding_config),
                speculator,
                enable_thinking=not no_thinking,
            )


@app.command(help="Classify text with a converted classifier model.")
def classify(
    model_path: Annotated[
        Path,
        Argument(
            help="Path to the classifier model directory.",
            metavar="MODEL_PATH",
        ),
    ],
    text: Annotated[
        str,
        Argument(
            help="Text to classify.",
            metavar="TEXT",
        ),
    ],
) -> None:
    model = ClassifierModel.load(model_path, ShardingConfig.replicated())
    scores = model.classify_chat(
        [UserMessage(text)],
        keychain=Keychain.init(0, sharding_config=model.sharding_config),
    )
    for label, score in scores.items():
        console.print(f"{label}: {score}")


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
class CliSpeculatorConversionCallbacks(SpeculatorConversionCallbacks):
    overwrite: bool = False

    stack: ExitStack = field(default_factory=ExitStack)
    progress: Progress | None = None
    loading_task: TaskID | None = None
    saving_task: TaskID | None = None

    def started(self) -> None:
        conversion_strs = [
            f"🚀 Converting DFlash speculator from [cyan]{self.hf_model_dir}[/cyan]",
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

    def loading_model(self) -> None:
        assert self.progress is not None

        self.loading_task = self.progress.add_task("Initializing DFlash speculator...")

    def finished_loading_model(self) -> None:
        assert self.progress is not None
        assert self.loading_task is not None

        self.progress.remove_task(self.loading_task)

    def saving_model(self) -> None:
        assert self.progress is not None

        self.saving_task = self.progress.add_task(f"💾 Saving the speculator to {self.output_dir}")

    def finished_saving_model(self) -> None:
        assert self.progress is not None
        assert self.saving_task is not None

        self.progress.remove_task(self.saving_task)
        self.stack.close()
        console.print(f"🧑‍🍳 Speculator successfully cooked and saved to [cyan]`{self.output_dir}`[/cyan]!")


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
    model = TTSModel.load(model_path, ShardingConfig.replicated())

    keychain = Keychain.init(0, sharding_config=model.sharding_config)
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
            show_default="bfloat16",
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


@speculator_app.command("convert", help="Import and export a DFlash speculator into the local Lalamo format.")
def speculator_convert(
    hf_model_dir: Annotated[
        Path,
        Argument(
            help="Path to a Hugging Face DFlash model directory.",
            metavar="HF_MODEL_DIR",
        ),
    ],
    dtype: Annotated[
        DType | None,
        Option(
            help="Dtype to use for activations and non-quantized weights.",
            show_default="bfloat16",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        Option(
            help="Directory to save the converted speculator to.",
            show_default="Saves the converted speculator in the `models/<hf_model_dir_name>` directory",
        ),
    ] = None,
    context_length: Annotated[
        int | None,
        Option(
            help="Maximum supported context length. Used to configure RoPE.",
            show_default="DFlash model's native maximum context length.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Overwrite existing speculator files.",
        ),
    ] = False,
) -> None:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / hf_model_dir.name

    _convert_speculator(
        hf_model_dir,
        output_dir,
        dtype,
        context_length,
        partial(CliSpeculatorConversionCallbacks, overwrite=overwrite),
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
        output_dir = DEFAULT_OUTPUT_DIR / PurePosixPath(model_spec.artifact_repo_id).name

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
