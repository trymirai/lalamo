from importlib.util import find_spec
from pathlib import Path
from typing import Annotated

import soundfile as sf
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer import Argument, Exit, Option, Typer

from lalamo.audio.utils import play_mono_audio
from lalamo.cli.common import console, err_console
from lalamo.message_processor import UserMessage
from lalamo.models import ClassifierModelConfig, LanguageModelConfig
from lalamo.models.tts_model import TTSGenerator, TTSMessage

app = Typer()


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
            for token in model.stream_reply_text(messages):
                console.print(token, end="")
                model_response_tokens.append(token)
            console.print()
            model_response_text = "".join(model_response_tokens)
            messages.append(model.message_processor.parse_response(model_response_text))
    else:
        for token in model.stream_reply_text([UserMessage(message)]):
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
