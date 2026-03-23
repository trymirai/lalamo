from __future__ import annotations

import re
import shutil
import tempfile
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
from tests.model_test_tiers import TIER_BY_REPO, ModelTier
from tests.resource_slots import resource_slots

# Keep this explicit. "default" is not the same as leaving the setting unset:
# unset lets JAX pick backend-specific behavior ("auto"), which can route to
# different kernels.
# We also observed that `high`/`highest` can trigger different GPU compile/fusion
# paths and produce much larger chunked-vs-unchunked numerical deltas in tests.
# Be careful when raising this precision for correctness baselines.
jax.config.update("jax_default_matmul_precision", "default")

GPU_ATOL = 1e-3
GPU_RTOL = 0.03


def _specs_up_to_tier(specs: tuple[ModelSpec, ...], tier: ModelTier) -> tuple[ModelSpec, ...]:
    return tuple(spec for spec in specs if TIER_BY_REPO.get(spec.repo, ModelTier.EXTRA) <= tier)


@pytest.fixture(autouse=True)
def _resource_slots(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None]:
    yield from resource_slots(request, tmp_path_factory)


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

CORE_LLM_SPECS: tuple[ModelSpec, ...] = _specs_up_to_tier(LLM_SPECS, ModelTier.CORE)
STANDARD_LLM_SPECS: tuple[ModelSpec, ...] = _specs_up_to_tier(LLM_SPECS, ModelTier.STANDARD)
EXTRA_LLM_SPECS: tuple[ModelSpec, ...] = _specs_up_to_tier(LLM_SPECS, ModelTier.EXTRA)


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


@pytest.fixture
def convert_model(
    model_registry: ModelRegistry,
) -> Generator[ConvertModel, None, None]:
    # Function-scoped cleanup is mandatory: without it the suite accumulates 2TB+ of converted weights.
    output_dirs: list[Path] = []

    def _convert(repo: str) -> Path:
        output_dir = Path(tempfile.mkdtemp()) / repo.replace("/", "__")
        convert(model_registry.repo_to_model[repo], output_dir)
        output_dirs.append(output_dir.parent)
        return output_dir

    yield _convert

    for d in output_dirs:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(params=TTS_SPECS, ids=[spec.repo for spec in TTS_SPECS])
def tts_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


@pytest.fixture(params=CORE_LLM_SPECS, ids=[spec.repo for spec in CORE_LLM_SPECS])
def core_llm_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


@pytest.fixture(params=STANDARD_LLM_SPECS, ids=[spec.repo for spec in STANDARD_LLM_SPECS])
def standard_llm_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


@pytest.fixture(params=EXTRA_LLM_SPECS, ids=[spec.repo for spec in EXTRA_LLM_SPECS])
def extra_llm_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param
