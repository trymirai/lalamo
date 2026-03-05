from pathlib import Path
from typing import Annotated

import jax.profiler
from typer import Context, Option, Typer

from lalamo.cli import chat, convert, generate, speculator
from lalamo.cli.common import console

app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

app.add_typer(chat.app)
app.add_typer(convert.app)
app.add_typer(generate.app)
app.add_typer(speculator.app, name="speculator", help="Train a speculator for a model.")


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
