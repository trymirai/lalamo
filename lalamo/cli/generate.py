from itertools import chain
from pathlib import Path
from typing import Annotated

import polars as pl
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from typer import Argument, Exit, Option, Typer

from lalamo.cli.common import console, err_console
from lalamo.common import flatten_parameters, get_default_device_bytes
from lalamo.data import load_hf_parquet
from lalamo.data.huggingface_message import HFMessage
from lalamo.message_processor import Message, UserMessage
from lalamo.models import LanguageModelConfig
from lalamo.models.common import BatchSizesComputedEvent, InferenceConfig
from lalamo.safetensors import safe_write

app = Typer()


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

    total_rows = pl.scan_parquet(dataset_path).select(pl.len()).collect().item()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("🧠 [cyan]Loading model...[/cyan]", total=None)
        model = LanguageModelConfig.load_model(model_path)
        progress.remove_task(task)

        task = progress.add_task("🗂️ [cyan]Loading dataset...[/cyan]", total=None)
        dataframe = load_hf_parquet(dataset_path).collect()
        conversations = dataframe.get_column("conversation")
        dataset = iter(
            [HFMessage.from_dict(message).as_message() for message in conversation] for conversation in conversations
        )
        try:
            first_row = next(dataset)
        except StopIteration:
            progress.remove_task(task)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"response": [], "chain_of_thought": []}).write_parquet(output_path)
            return
        dataset = chain([first_row], dataset)
        progress.remove_task(task)

        estimating_task = progress.add_task("📐 [cyan]Estimating the best batch sizes...[/cyan]", total=None)
        generation_task: TaskID | None = None

        def on_batch_sizes_computed(event: BatchSizesComputedEvent) -> None:
            nonlocal generation_task
            progress.remove_task(estimating_task)
            for info in event.batch_sizes:
                progress.console.print(
                    f"Prefix length {info.prefix_length} has {info.num_elements} elements, "
                    f"with batchsize of {info.batch_size}",
                )
            generation_task = progress.add_task("🔮 [cyan]Generating replies...[/cyan]", total=total_rows)

        inference_config = InferenceConfig(max_output_length=max_output_length, batch_size=batch_size)

        replies = []
        for rows_processed, (idx, reply) in enumerate(
            model.reply_many(
                dataset,
                inference_config=inference_config,
                vram_bytes=max_vram,
                batch_sizes_callback=on_batch_sizes_computed,
            ),
        ):
            replies.append((idx, reply))
            if generation_task is not None:
                progress.update(generation_task, completed=rows_processed + 1)

    replies.sort(key=lambda x: x[0])

    df = pl.DataFrame(
        {
            "response": [reply.response for _, reply in replies],
            "chain_of_thought": [reply.chain_of_thought for _, reply in replies],
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    console.print(f"💾 Replies saved to [cyan]{output_path}[/cyan]")


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

    if output_path.exists():
        if not overwrite and not Confirm().ask(
            rf"⚠️ Output [cyan]{output_path}[/cyan] already exists."
            r" Do you want to overwrite it?",
        ):
            raise Exit
        output_path.unlink()

    messages: list[Message] | None = None if message is None else [UserMessage(content=message)]

    console.print(f"🔍 Tracing [cyan]{model_path}[/cyan]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("🧠 Loading model...")
        model = LanguageModelConfig.load_model(model_path)
        progress.update(task, description="🔍 Recording trace...")
        result = model.record_trace(messages)
        progress.update(task, description=f"💾 Saving trace to {output_path}")
        traces = flatten_parameters(result.export())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(output_path).open("wb") as fd:
            safe_write(fd, traces)
    console.print(f"💾 Trace saved to [cyan]{output_path}[/cyan]")
