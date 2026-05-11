import functools
import json
import tarfile
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import huggingface_hub
import jax.numpy as jnp
import yaml

from lalamo.safetensors import safe_read
from lalamo.utils.lazy_collections import MapDictValues
from lalamo.utils.registry_abc import RegistryABC

type WeightShard = tuple[Mapping[str, jnp.ndarray], Mapping[str, str]]
type WeightShardContext = AbstractContextManager[WeightShard]


class WeightFormat(StrEnum):
    SAFETENSORS = ".safetensors"
    TORCH = ".pth"


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


def report_status(progress_callback: Callable[[StatusEvent], None] | None, event: StatusEvent) -> None:
    if progress_callback is not None:
        progress_callback(event)


@contextmanager
def load_torch_weights(path: Path, *, weights_only: bool = True) -> Generator[WeightShard]:
    import torch  # noqa: PLC0415

    from lalamo.utils.torch_interop import torch_to_jax  # noqa: PLC0415

    torch_weights = torch.load(path, map_location="cpu", weights_only=weights_only)
    yield MapDictValues(torch_to_jax, torch_weights), {}


@contextmanager
def load_safetensors_weights(path: Path) -> Generator[WeightShard]:
    with path.open("rb") as fd:
        metadata, weights = safe_read(fd)
        yield weights, metadata or {}


class Origin(RegistryABC):
    @abstractmethod
    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Path: ...

    @abstractmethod
    def get_weights(
        self,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShardContext]: ...

    @property
    @abstractmethod
    def description(self) -> str: ...


def hf_resolve_file(
    repo: str,
    file_spec: FileSpec,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> Path:
    report_status(progress_callback, DownloadingFileEvent(file_spec))
    result = huggingface_hub.hf_hub_download(
        repo_id=file_spec.repo or repo,
        filename=file_spec.filename,
    )
    report_status(progress_callback, FinishedDownloadingFileEvent(file_spec))
    return Path(result)


def hf_resolve_weights(
    repo: str,
    extension: str,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[Path, ...]:
    all_files = huggingface_hub.list_repo_files(repo)
    return tuple(
        hf_resolve_file(repo, FileSpec(filename), progress_callback)
        for filename in all_files
        if filename.endswith(extension)
    )


def load_weight_paths(paths: Sequence[Path], weight_format: WeightFormat) -> tuple[WeightShardContext, ...]:
    match weight_format:
        case WeightFormat.SAFETENSORS:
            return tuple(load_safetensors_weights(path) for path in paths)
        case WeightFormat.TORCH:
            return tuple(load_torch_weights(path) for path in paths)


@dataclass(frozen=True)
class HuggingFaceOrigin(Origin):
    repo: str
    weight_format: WeightFormat = WeightFormat.SAFETENSORS

    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Path:
        return hf_resolve_file(self.repo, file_spec, progress_callback)

    def get_weights(
        self,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShardContext]:
        paths = hf_resolve_weights(self.repo, self.weight_format.value, progress_callback)
        return load_weight_paths(paths, self.weight_format)

    @property
    def description(self) -> str:
        return self.repo


@dataclass(frozen=True)
class NemoOrigin(Origin):
    repo: str

    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Path:
        if file_spec.filename.endswith(".nemo"):
            (nemo_path,) = hf_resolve_weights(self.repo, ".nemo", progress_callback)
            _, config_path = self.extract_nemo_archive(nemo_path)
            return config_path
        return hf_resolve_file(self.repo, file_spec, progress_callback)

    def get_weights(
        self,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShardContext]:
        (nemo_path,) = hf_resolve_weights(self.repo, ".nemo", progress_callback)
        weight_paths, _ = self.extract_nemo_archive(nemo_path)
        return tuple(load_torch_weights(path) for path in weight_paths)

    @property
    def description(self) -> str:
        return self.repo

    @classmethod
    @functools.cache
    def extract_nemo_archive(cls, nemo_path: Path) -> tuple[tuple[Path, ...], Path]:
        tmpdir = Path(tempfile.mkdtemp())
        with tarfile.open(nemo_path, "r") as tar:
            for tar_member in tar.getmembers():
                if not (
                    tar_member.name.startswith("..") or Path(tar_member.name).is_absolute() or tar_member.size == 0
                ):
                    tar.extract(tar_member.name, path=tmpdir)

        weights_paths = tuple(tmpdir.glob("*.ckpt"))
        if not weights_paths:
            raise FileNotFoundError("Failed to find Nemo model weights")
        (yaml_config_path,) = tmpdir.glob("*.yaml")

        with yaml_config_path.open() as f:
            config_yaml = yaml.safe_load(f)
        config_path = yaml_config_path.with_suffix(".json")
        with config_path.open("w") as f:
            json.dump(config_yaml, f)

        return weights_paths, config_path


@dataclass(frozen=True)
class LocalOrigin(Origin):
    root: str
    weight_files: tuple[str, ...] = ()
    weight_format: WeightFormat = WeightFormat.SAFETENSORS

    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,  # noqa: ARG002
    ) -> Path:
        return Path(self.root) / file_spec.filename

    def get_weights(
        self,
        progress_callback: Callable[[StatusEvent], None] | None = None,  # noqa: ARG002
    ) -> Sequence[WeightShardContext]:
        paths = tuple(Path(self.root) / weight_file for weight_file in self.weight_files)
        return load_weight_paths(paths, self.weight_format)

    @property
    def description(self) -> str:
        return self.root
