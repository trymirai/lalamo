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
from tests.model_test_tiers import TIER_BY_REPO, ModelSize, ModelTier, model_size

# Keep this explicit. "default" is not the same as leaving the setting unset:
# unset lets JAX pick backend-specific behavior ("auto"), which can route to
# different kernels.
# We also observed that `high`/`highest` can trigger different GPU compile/fusion
# paths and produce much larger chunked-vs-unchunked numerical deltas in tests.
# Be careful when raising this precision for correctness baselines.
jax.config.update("jax_default_matmul_precision", "default")

FAST_MARKER = pytest.mark.fast


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(FAST_MARKER)


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
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._cache: dict[str, Path] = {}
        self._local_dirs: list[Path] = []

    def _convert(self, repo: str) -> Path:
        output_dir = Path(tempfile.mkdtemp()) / repo.replace("/", "__")
        convert(self._registry.repo_to_model[repo], output_dir)
        return output_dir

    def __call__(self, repo: str, *, cached: bool = False) -> Path:
        if cached:
            if repo not in self._cache:
                self._cache[repo] = self._convert(repo)
            return self._cache[repo]

        output_dir = self._convert(repo)
        self._local_dirs.append(output_dir.parent)
        return output_dir

    def cleanup_local(self) -> None:
        for local_dir in self._local_dirs:
            shutil.rmtree(local_dir, ignore_errors=True)
        self._local_dirs.clear()

    def cleanup_all(self) -> None:
        self.cleanup_local()
        for output_dir in self._cache.values():
            shutil.rmtree(output_dir.parent, ignore_errors=True)
        self._cache.clear()


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
def _convert_model_session(model_registry: ModelRegistry) -> Generator[ConvertModel, None, None]:
    converter = ConvertModel(model_registry)
    yield converter
    converter.cleanup_all()


@pytest.fixture
def convert_model(_convert_model_session: ConvertModel) -> Generator[ConvertModel, None, None]:
    yield _convert_model_session
    _convert_model_session.cleanup_local()
