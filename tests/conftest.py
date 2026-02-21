from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import jax
import pytest
from typer.testing import CliRunner

from lalamo.commands import convert
from lalamo.main import app
from lalamo.model_import.model_configs.huggingface import HuggingFaceLMConfig
from lalamo.model_import.model_specs.common import ModelSpec, ModelType
from lalamo.model_registry import ModelRegistry

# Keep this explicit. "default" is not the same as leaving the setting unset:
# unset lets JAX pick backend-specific behavior ("auto"), which can route to
# different kernels.
# We also observed that `high`/`highest` can trigger different GPU compile/fusion
# paths and produce much larger chunked-vs-unchunked numerical deltas in tests.
# Be careful when raising this precision for correctness baselines.
jax.config.update("jax_default_matmul_precision", "default")

RunLalamo = Callable[..., str]
ConvertModel = Callable[[str], Path]

ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

HF_MODEL_SPECS: tuple[ModelSpec, ...] = tuple(
    spec
    for spec in ModelRegistry.build(allow_third_party_plugins=False).models
    if issubclass(spec.config_type, HuggingFaceLMConfig)
)

HF_LANGUAGE_MODEL_REPOS: tuple[str, ...] = tuple(
    spec.repo
    for spec in HF_MODEL_SPECS
    if spec.model_type == ModelType.LANGUAGE_MODEL
)


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
def model_registry() -> ModelRegistry:
    return ModelRegistry.build(allow_third_party_plugins=False)


@pytest.fixture(scope="session")
def convert_model(
    model_registry: ModelRegistry,
    tmp_path_factory: pytest.TempPathFactory,
) -> ConvertModel:
    def _convert(repo: str) -> Path:
        output_dir = tmp_path_factory.getbasetemp() / "converted_models" / repo.replace("/", "__")
        if not (output_dir / "config.json").exists():
            convert(model_registry.repo_to_model[repo], output_dir)
        return output_dir

    return _convert


@pytest.fixture(params=HF_MODEL_SPECS, ids=[spec.repo for spec in HF_MODEL_SPECS])
def hf_model_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param
