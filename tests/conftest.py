from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lalamo.main import app

from lalamo.commands import convert
from lalamo.model_import import REPO_TO_MODEL

RunLalamo = Callable[..., str]
ConvertModel = Callable[[str], Path]


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
