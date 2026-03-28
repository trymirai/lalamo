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
from tests.model_test_tiers import COHERENCE_TTS_REPOS, TIER_BY_REPO, ModelSize, ModelTier, model_size

# Keep this explicit. "default" is not the same as leaving the setting unset:
# unset lets JAX pick backend-specific behavior ("auto"), which can route to
# different kernels.
# We also observed that `high`/`highest` can trigger different GPU compile/fusion
# paths and produce much larger chunked-vs-unchunked numerical deltas in tests.
# Be careful when raising this precision for correctness baselines.
jax.config.update("jax_default_matmul_precision", "default")

GPU_ATOL = 1e-3
GPU_RTOL = 0.03

ALL_MODEL_SPECS: tuple[ModelSpec, ...] = ModelRegistry.build(allow_third_party_plugins=False).models


def filter_specs(
    *,
    model_type: ModelType | None = None,
    max_tier: ModelTier | None = None,
    repos: frozenset[str] | None = None,
) -> tuple[ModelSpec, ...]:
    specs = ALL_MODEL_SPECS
    if model_type is not None:
        specs = tuple(spec for spec in specs if spec.model_type == model_type)
    if max_tier is not None:
        specs = tuple(spec for spec in specs if TIER_BY_REPO.get(spec.repo, ModelTier.EXTRA) <= max_tier)
    if repos is not None:
        specs = tuple(spec for spec in specs if spec.repo in repos)
    return specs


SIZE_MARKS: dict[ModelSize, pytest.MarkDecorator] = {
    ModelSize.SMALL: pytest.mark.small_model,
    ModelSize.LARGE: pytest.mark.large_model,
}


def mark_by_size(specs: tuple[ModelSpec, ...]) -> list[ModelSpec | pytest.param]:
    return [pytest.param(spec, marks=SIZE_MARKS[model_size(spec)]) for spec in specs]


@pytest.fixture(autouse=True)
def _gpu_tolerance() -> Generator[None]:
    if any(device.platform == "gpu" for device in jax.devices()):
        with tolerance(atol=GPU_ATOL, rtol=GPU_RTOL):
            yield
    else:
        yield


RunLalamo = Callable[..., str]


class ConvertModel:
    def __init__(
        self,
        registry: ModelRegistry,
        cache: dict[str, Path],
        cache_dirs: list[Path],
    ) -> None:
        self._registry = registry
        self._cache = cache
        self._cache_dirs = cache_dirs
        self._local_dirs: list[Path] = []

    def __call__(self, repo: str, *, cached: bool = False) -> Path:
        if cached:
            if repo not in self._cache:
                output_dir = Path(tempfile.mkdtemp()) / repo.replace("/", "__")
                convert(self._registry.repo_to_model[repo], output_dir)
                self._cache[repo] = output_dir
                self._cache_dirs.append(output_dir.parent)
            return self._cache[repo]

        output_dir = Path(tempfile.mkdtemp()) / repo.replace("/", "__")
        convert(self._registry.repo_to_model[repo], output_dir)
        self._local_dirs.append(output_dir.parent)
        return output_dir

    def cleanup_local(self) -> None:
        for local_dir in self._local_dirs:
            shutil.rmtree(local_dir, ignore_errors=True)
        self._local_dirs.clear()


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
def model_registry() -> ModelRegistry:
    return ModelRegistry.build(allow_third_party_plugins=False)


@pytest.fixture(scope="session")
def _convert_model_cache(
    model_registry: ModelRegistry,
) -> Generator[tuple[dict[str, Path], list[Path]], None, None]:
    cache: dict[str, Path] = {}
    cache_dirs: list[Path] = []
    yield cache, cache_dirs
    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def convert_model(
    model_registry: ModelRegistry,
    _convert_model_cache: tuple[dict[str, Path], list[Path]],
) -> Generator[ConvertModel, None, None]:
    cache, cache_dirs = _convert_model_cache
    converter = ConvertModel(model_registry, cache, cache_dirs)
    yield converter
    converter.cleanup_local()


tts_specs = filter_specs(model_type=ModelType.TTS_MODEL)


@pytest.fixture(params=mark_by_size(tts_specs), ids=[spec.repo for spec in tts_specs])
def tts_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


coherence_tts_specs = filter_specs(model_type=ModelType.TTS_MODEL, repos=frozenset(COHERENCE_TTS_REPOS))


@pytest.fixture(params=coherence_tts_specs, ids=[spec.repo for spec in coherence_tts_specs])
def coherence_tts_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


core_llm_specs = filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CORE)


@pytest.fixture(params=mark_by_size(core_llm_specs), ids=[spec.repo for spec in core_llm_specs])
def core_llm_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


standard_llm_specs = filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.STANDARD)


@pytest.fixture(params=mark_by_size(standard_llm_specs), ids=[spec.repo for spec in standard_llm_specs])
def standard_llm_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


extra_llm_specs = filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.EXTRA)


@pytest.fixture(params=mark_by_size(extra_llm_specs), ids=[spec.repo for spec in extra_llm_specs])
def extra_llm_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param
