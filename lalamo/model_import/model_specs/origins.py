import functools
import json
import tarfile
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import huggingface_hub
import jax.numpy as jnp
from jaxtyping import DTypeLike

from lalamo.common import cast_if_float
from lalamo.registry_abc import RegistryABC
from lalamo.safetensors import safe_read
from lalamo.utils import MapDictValues


@dataclass(frozen=True)
class FileSpec:
    filename: str
    repo: str | None = None


class DownloadingFileEvent(NamedTuple):
    file: FileSpec


class FinishedDownloadingFileEvent(NamedTuple):
    file: FileSpec


class InitializingModelEvent(NamedTuple):
    pass


class FinishedInitializingModelEvent(NamedTuple):
    pass


type StatusEvent = (
    DownloadingFileEvent | FinishedDownloadingFileEvent | InitializingModelEvent | FinishedInitializingModelEvent
)


class Origin(RegistryABC):
    @abstractmethod
    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Path: ...

    @abstractmethod
    def resolve_weights(
        self,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> list[Path]: ...

    @abstractmethod
    @contextmanager
    def load_weights(
        self,
        path: Path,
        float_dtype: DTypeLike,
    ) -> Iterator[tuple[Mapping[str, jnp.ndarray], Mapping[str, str]]]: ...

    @property
    @abstractmethod
    def description(self) -> str: ...


def hf_resolve_file(
    repo: str,
    file_spec: FileSpec,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> Path:
    if progress_callback is not None:
        progress_callback(DownloadingFileEvent(file_spec))
    result = huggingface_hub.hf_hub_download(
        repo_id=file_spec.repo or repo,
        filename=file_spec.filename,
    )
    if progress_callback is not None:
        progress_callback(FinishedDownloadingFileEvent(file_spec))
    return Path(result)


def hf_resolve_weights(
    repo: str,
    extension: str,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> list[Path]:
    all_files = huggingface_hub.list_repo_files(repo)
    return [hf_resolve_file(repo, FileSpec(f), progress_callback) for f in all_files if f.endswith(extension)]


@dataclass(frozen=True)
class HuggingFaceOrigin(Origin):
    repo: str

    def resolve_file(
        self, file_spec: FileSpec, progress_callback: Callable[[StatusEvent], None] | None = None
    ) -> Path:
        return hf_resolve_file(self.repo, file_spec, progress_callback)

    def resolve_weights(self, progress_callback: Callable[[StatusEvent], None] | None = None) -> list[Path]:
        return hf_resolve_weights(self.repo, ".safetensors", progress_callback)

    @contextmanager
    def load_weights(
        self,
        path: Path,
        float_dtype: DTypeLike,
    ) -> Iterator[tuple[Mapping[str, jnp.ndarray], Mapping[str, str]]]:
        with path.open("rb") as fd:
            metadata, weights = safe_read(fd)
            yield MapDictValues(lambda v: cast_if_float(v, float_dtype), weights), metadata or {}

    @property
    def description(self) -> str:
        return self.repo


@dataclass(frozen=True)
class HuggingFaceTorchOrigin(Origin):
    repo: str

    def resolve_file(
        self, file_spec: FileSpec, progress_callback: Callable[[StatusEvent], None] | None = None
    ) -> Path:
        return hf_resolve_file(self.repo, file_spec, progress_callback)

    def resolve_weights(self, progress_callback: Callable[[StatusEvent], None] | None = None) -> list[Path]:
        return hf_resolve_weights(self.repo, ".pth", progress_callback)

    @contextmanager
    def load_weights(
        self,
        path: Path,
        float_dtype: DTypeLike,
    ) -> Iterator[tuple[Mapping[str, jnp.ndarray], Mapping[str, str]]]:
        import torch

        from lalamo.modules.torch_interop import torch_to_jax

        torch_weights = torch.load(path, map_location="cpu", weights_only=True)
        yield MapDictValues(lambda v: cast_if_float(torch_to_jax(v), float_dtype), torch_weights), {}

    @property
    def description(self) -> str:
        return self.repo


@functools.cache
def extract_nemo_archive(nemo_path: Path) -> tuple[tuple[Path, ...], Path]:
    import yaml

    tmpdir = Path(tempfile.mkdtemp())
    with tarfile.open(nemo_path, "r") as tar:
        for member in tar.getmembers():
            if not (member.name.startswith("..") or Path(member.name).is_absolute() or member.size == 0):
                tar.extract(member.name, path=tmpdir)

    weights_paths = tuple(tmpdir.glob("*.ckpt"))
    if not weights_paths:
        raise FileNotFoundError("Failed to find Nemo model weights")
    (yaml_path,) = list(tmpdir.glob("*.yaml"))

    with open(yaml_path) as f:
        config_yaml = yaml.safe_load(f)
    config_json_path = yaml_path.with_suffix(".json")
    with open(config_json_path, "w") as f:
        json.dump(config_yaml, f)

    return weights_paths, config_json_path


@dataclass(frozen=True)
class NemoOrigin(Origin):
    """NVIDIA NeMo models: .nemo tar archives from HuggingFace containing torch checkpoints."""

    repo: str

    def resolve_file(
        self, file_spec: FileSpec, progress_callback: Callable[[StatusEvent], None] | None = None
    ) -> Path:
        if file_spec.filename.endswith(".nemo"):
            (nemo_path,) = hf_resolve_weights(self.repo, ".nemo", progress_callback)
            _, config_path = extract_nemo_archive(nemo_path)
            return config_path
        return hf_resolve_file(self.repo, file_spec, progress_callback)

    def resolve_weights(self, progress_callback: Callable[[StatusEvent], None] | None = None) -> list[Path]:
        (nemo_path,) = hf_resolve_weights(self.repo, ".nemo", progress_callback)
        weights, _ = extract_nemo_archive(nemo_path)
        return list(weights)

    @contextmanager
    def load_weights(
        self,
        path: Path,
        float_dtype: DTypeLike,
    ) -> Iterator[tuple[Mapping[str, jnp.ndarray], Mapping[str, str]]]:
        import torch

        from lalamo.modules.torch_interop import torch_to_jax

        torch_weights = torch.load(path, map_location="cpu", weights_only=True)
        yield MapDictValues(lambda v: cast_if_float(torch_to_jax(v), float_dtype), torch_weights), {}

    @property
    def description(self) -> str:
        return self.repo
