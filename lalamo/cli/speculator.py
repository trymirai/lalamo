import random
from itertools import chain, islice
from pathlib import Path
from typing import Annotated

from rich import box
from rich.live import Live
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from typer import Argument, Exit, Option, Typer

from lalamo.cli.common import console, err_console
from lalamo.common import get_default_device_bytes, get_usable_memory_from_bytes
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.models import LanguageModelConfig
from lalamo.models.common import InferenceConfig
from lalamo.models.lm_helpers import estimate_batchsize_from_bytes
from lalamo.speculator.inference import inference_collect_traces
from lalamo.speculator.ngram import NGramSpeculator
from lalamo.speculator.utils import test_speculator, train_speculator

app = Typer()


@app.command(help="Estimate maximum batch size at which a model can be run.")
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
        mem_bytes = vram_gb * 1000 * 1000 * 1000
    elif (mem_bytes := get_default_device_bytes()) is None:
        err_console.print("Cannot get the default device's memory stats, use --vram-gb")
        raise Exit(1)

    usable_mem = get_usable_memory_from_bytes(mem_bytes)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Loading model...[/cyan]")
        model = LanguageModelConfig.load_model(model_path)
        progress.remove_task(task)

        estimating_task = progress.add_task("[cyan]Estimating batch size...[/cyan]")

        def memory_per_batchsize(batch_size: int) -> int:
            inference_config = InferenceConfig(
                max_output_length=max_output_length,
                padded_length=max_input_length,
                num_top_logits_to_return=num_logits_per_token,
                batch_size=batch_size,
            )
            return model.estimate_memory_consumption(inference_config=inference_config)

        def on_estimate_progress(event: object) -> None:
            hi_str = str(event.hi) if event.hi is not None else "?"
            desc = f"[cyan]Estimating batch size... ({event.lo}..{hi_str})[/cyan]"
            progress.update(estimating_task, description=desc)

        bs = estimate_batchsize_from_bytes(memory_per_batchsize, usable_mem, on_estimate_progress)
        progress.remove_task(estimating_task)

    console.print(f"Found maximum batch size: [cyan]{bs}[/cyan]")


@app.command(help="Run model inference and collect traces for speculator training")
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
        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True)
        live.update(progress, refresh=True)

        task = progress.add_task("🧠 [cyan]Loading model...[/cyan]")
        model = LanguageModelConfig.load_model(model_path)
        progress.remove_task(task)

        task = progress.add_task("🗂️ [cyan]Loading dataset...[/cyan]")
        dataframe = shuffle_dataset(load_hf_parquet(dataset_path))
        conversations = dataframe.get_column("conversation")
        dataset = iter(
            [HFMessage.from_dict(message).as_message() for message in conversation] for conversation in conversations
        )
        dataset = chain([next(dataset)], dataset)
        progress.remove_task(task)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        live.update(progress, refresh=True)
        inference_task = progress.add_task(
            "🔮 [cyan]Running inference...[/cyan]",
            total=num_tokens_to_generate,
        )

        traces = inference_collect_traces(
            model,
            dataset,
            num_logits_per_token,
            batch_size,
            max_input_length,
            max_output_length,
            num_tokens_to_generate,
            lambda event: progress.update(inference_task, completed=event.tokens_generated),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as output_fd:
            output_fd.writelines(trace_item.serialize() for trace_item in traces)

        progress.update(inference_task, description="✅ Completed")


@app.command(help="View model inference traces")
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

        table = Table(show_lines=True, box=box.ROUNDED)
        table.add_column("Prefix")
        table.add_column("Completion")

        for completion in islice(traces, num_completions):
            detokenized_prefix = model.message_processor.detokenize(completion.prefix_token_ids)
            detokenized_completion = model.message_processor.detokenize(completion.completion_token_ids)
            table.add_row(Text(detokenized_prefix), Text(detokenized_completion))

        console.print(table)


@app.command(help="Train a speculator from inference traces")
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
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        training_task = progress.add_task("🔮 [cyan]Training speculator...[/cyan]", total=subsample_size)

        with open(trace_path, "rb") as trace_fd:
            traces = LalamoCompletion.deserialize_many(trace_fd)
            speculator = NGramSpeculator.new(hashtable_size, num_logits_per_token, ngram_size)
            train_speculator(
                speculator,
                traces,
                subsample_size,
                lambda event: progress.update(training_task, completed=event.trained_tokens),
            )

        progress.update(training_task, description="✅ Completed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fd:
        fd.write(speculator.serialize())
    console.print(f"💾 Speculator saved to [cyan]{output_path}[/cyan]")


@app.command(help="Run speculator as an autoregressive llm")
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

    table = Table(show_header=False, show_lines=True, box=box.ROUNDED)

    if seed is not None:
        random.seed(seed)

    for _ in range(num_sequences):
        sequence = test_speculator(speculator)
        detokenized = model.message_processor.detokenize(sequence)
        table.add_row(detokenized)

    console.print(table)
