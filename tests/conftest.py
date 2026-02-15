from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import jax
import pytest
from typer.testing import CliRunner

# Keep this explicit. "default" is not the same as leaving the setting unset:
# unset lets JAX pick backend-specific behavior ("auto"), which can route to
# different kernels.
# We also observed that `high`/`highest` can trigger different GPU compile/fusion
# paths and produce much larger chunked-vs-unchunked numerical deltas in tests.
# Be careful when raising this precision for correctness baselines.
jax.config.update("jax_default_matmul_precision", "default")

from lalamo.commands import convert
from lalamo.main import app
from lalamo.model_import import REPO_TO_MODEL

RunLalamo = Callable[..., str]
ConvertModel = Callable[[str], Path]

ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi_escape(text: str) -> str:
    return ANSI_ESCAPE_REGEX.sub("", text)


@pytest.fixture(scope="session")
def run_lalamo() -> RunLalamo:
    runner = CliRunner()

    def _run(*args: str) -> str:
        result = runner.invoke(app, list(args), terminal_width=240)
        assert result.exit_code == 0, (
            f"lalamo {' '.join(args)} failed (exit {result.exit_code}).\n"
            f"--- output ---\n{result.output}\n"
            f"--- exception ---\n{result.exception!r}"
        )
        return result.output

    return _run


@pytest.fixture(scope="session")
def convert_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> ConvertModel:
    def _convert(repo: str) -> Path:
        output_dir = tmp_path_factory.getbasetemp() / "converted_models" / repo.replace("/", "__")
        if not (output_dir / "config.json").exists():
            convert(REPO_TO_MODEL[repo], output_dir)
        return output_dir

    return _convert
