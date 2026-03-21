from __future__ import annotations

import re
from collections.abc import Callable, Generator
from pathlib import Path

import jax
import pytest
from typer.testing import CliRunner

from lalamo.commands import convert
from lalamo.main import app
from lalamo.model_import.model_specs.common import ModelSpec, ModelType
from lalamo.model_registry import ModelRegistry
from tests.common import tolerance

# Keep this explicit. "default" is not the same as leaving the setting unset:
# unset lets JAX pick backend-specific behavior ("auto"), which can route to
# different kernels.
# We also observed that `high`/`highest` can trigger different GPU compile/fusion
# paths and produce much larger chunked-vs-unchunked numerical deltas in tests.
# Be careful when raising this precision for correctness baselines.
jax.config.update("jax_default_matmul_precision", "default")

GPU_ATOL = 1e-3
GPU_RTOL = 0.03


@pytest.fixture(autouse=True)
def _gpu_tolerance() -> Generator[None]:
    if any(device.platform == "gpu" for device in jax.devices()):
        with tolerance(atol=GPU_ATOL, rtol=GPU_RTOL):
            yield
    else:
        yield

RunLalamo = Callable[..., str]
ConvertModel = Callable[[str], Path]

ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

ALL_MODEL_SPECS: tuple[ModelSpec, ...] = ModelRegistry.build(allow_third_party_plugins=False).models

LLM_SPECS: tuple[ModelSpec, ...] = tuple(
    spec for spec in ALL_MODEL_SPECS if spec.model_type == ModelType.LANGUAGE_MODEL
)

TTS_SPECS: tuple[ModelSpec, ...] = tuple(
    spec for spec in ALL_MODEL_SPECS if spec.model_type == ModelType.TTS_MODEL
)

HF_LANGUAGE_MODEL_REPOS: tuple[str, ...] = tuple(spec.repo for spec in LLM_SPECS)


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


@pytest.fixture(params=ALL_MODEL_SPECS, ids=[spec.repo for spec in ALL_MODEL_SPECS])
def all_model_specs(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


@pytest.fixture(params=LLM_SPECS, ids=[spec.repo for spec in LLM_SPECS])
def llm_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


@pytest.fixture(params=TTS_SPECS, ids=[spec.repo for spec in TTS_SPECS])
def tts_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param
