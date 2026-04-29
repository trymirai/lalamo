# ruff: noqa: E402
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Generator, Mapping
from functools import cache
from os import environ
from pathlib import Path
from typing import Any, cast

from jax._src.test_util import request_cpu_devices

request_cpu_devices(8)

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from jax.sharding import AxisType, Mesh
from jaxtyping import DTypeLike
from tokenizers import Tokenizer
from typer.testing import CliRunner

from lalamo.commands import convert
from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.main import app
from lalamo.model import Model, ModelConfig
from lalamo.model_import.model_spec import ModelSpec
from lalamo.model_registry import ModelRegistry
from lalamo.module import ShardingAxis
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import WeightMatrix, WeightMatrixSpec
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
SKIP_REQUIRES_JAX_GPU_MARKER = pytest.mark.skip(reason="requires a JAX GPU device")
JAX_CUDA_BACKEND = "cuda"


def _requested_jax_platforms() -> frozenset[str]:
    jax_platforms = environ.get("JAX_PLATFORMS") or environ.get("JAX_PLATFORM_NAME")
    if jax_platforms is None:
        return frozenset()
    return frozenset(platform.strip().lower() for platform in jax_platforms.split(","))


def _is_gpu_backend_requested() -> bool:
    requested_platforms = _requested_jax_platforms()
    if requested_platforms:
        return JAX_CUDA_BACKEND in requested_platforms

    visible_cuda_devices = environ.get("CUDA_VISIBLE_DEVICES")
    return visible_cuda_devices not in {None, "", "-1"}


@cache
def _has_jax_gpu_device() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        if _is_gpu_backend_requested():
            raise
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    has_jax_gpu_device = _has_jax_gpu_device()

    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(FAST_MARKER)
        if item.get_closest_marker("gpu") is not None and not has_jax_gpu_device:
            item.add_marker(SKIP_REQUIRES_JAX_GPU_MARKER)


@pytest.fixture
def fake_mesh() -> Generator[Mesh]:
    mesh = jax.make_mesh(
        (2, 2, 2),
        (ShardingAxis.DATA, ShardingAxis.TENSOR, ShardingAxis.EXPERT),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        devices=jax.devices("cpu")[:8],
    )
    with jax.set_mesh(mesh):
        yield mesh


@pytest.fixture
def tracer_mesh() -> Generator[Mesh]:
    mesh = jax.make_mesh(
        (1, 2, 4),
        (ShardingAxis.DATA, ShardingAxis.TENSOR, ShardingAxis.EXPERT),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        devices=jax.devices("cpu")[:8],
    )
    with jax.set_mesh(mesh):
        yield mesh


GPU_ATOL = 1e-3
GPU_RTOL = 0.03

ALL_MODEL_SPECS: tuple[ModelSpec, ...] = ModelRegistry.build(allow_third_party_plugins=False).models


def filter_specs(
    *,
    model_type: type[ModelSpec] | tuple[type[ModelSpec], ...] | None = None,
    max_tier: ModelTier | None = None,
    repos: frozenset[str] | None = None,
) -> tuple[ModelSpec, ...]:
    specs = ALL_MODEL_SPECS
    if model_type is not None:
        specs = tuple(spec for spec in specs if isinstance(spec, model_type))
    if max_tier is not None:
        specs = tuple(spec for spec in specs if TIER_BY_REPO.get(spec.origin.description, ModelTier.EXTRA) <= max_tier)
    if repos is not None:
        specs = tuple(spec for spec in specs if spec.origin.description in repos)
    return specs


SIZE_MARKS: dict[ModelSize, pytest.MarkDecorator] = {
    ModelSize.SMALL: pytest.mark.small_model,
    ModelSize.LARGE: pytest.mark.large_model,
}


def mark_by_size(specs: tuple[ModelSpec, ...]) -> list[Any]:
    return [pytest.param(spec, marks=SIZE_MARKS[model_size(spec)]) for spec in specs]


def _decode_safetensors_metadata(metadata: dict[str, str] | None) -> dict[str, JSON]:
    if metadata is None:
        return {}
    return {key: json.loads(value) for key, value in metadata.items()}


def _infer_model_dtype(arrays: Mapping[str, jax.Array]) -> DTypeLike:
    for key in sorted(arrays):
        array_dtype = arrays[key].dtype
        if jnp.issubdtype(array_dtype, jnp.inexact):
            return array_dtype
    raise ValueError("Converted model does not contain any floating-point arrays.")


def _with_exported_weight_matrix_specs(model: Model, metadata: Mapping[str, JSON]) -> Model:
    def is_weight_matrix(value: object) -> bool:
        return isinstance(value, WeightMatrix)

    def align_weight_matrix(path: tuple[object, ...], value: object) -> object:
        if not isinstance(value, WeightMatrix):
            return value

        spec_json = metadata.get(ParameterPath() / path / "spec")
        if spec_json is None:
            return value

        exported_spec = WeightMatrixSpec.from_json(spec_json)
        if exported_spec == value.spec:
            return value
        return exported_spec.compress(value.decompress())

    return cast("Model", jtu.tree_map_with_path(align_weight_matrix, model, is_leaf=is_weight_matrix))


def load_converted_model(model_dir: Path) -> Model:
    with (model_dir / "config.json").open() as config_file:
        config = ModelConfig.from_json(json.load(config_file))
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    with (model_dir / "model.safetensors").open("rb") as weights_file:
        metadata, arrays = safe_read(weights_file)
        decoded_metadata = _decode_safetensors_metadata(metadata)
        model = config.init(tokenizer, EmptyInitializer(_infer_model_dtype(arrays)))
        model = _with_exported_weight_matrix_specs(model, decoded_metadata)
        return model.load_exported(
            ExportResults(
                arrays=arrays,
                metadata=decoded_metadata,
            ),
            allow_dtype_cast=True,
        )


@pytest.fixture(autouse=True)
def _gpu_tolerance() -> Generator[None]:
    if _has_jax_gpu_device():
        with tolerance(atol=GPU_ATOL, rtol=GPU_RTOL):
            yield
    else:
        yield


ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi_escape(text: str) -> str:
    return ANSI_ESCAPE_REGEX.sub("", text)


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
