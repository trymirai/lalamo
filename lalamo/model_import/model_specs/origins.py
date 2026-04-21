import functools
import json
import tarfile
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import huggingface_hub
from jaxtyping import DTypeLike
import yaml

from lalamo.common import WeightShard, cast_if_float
from lalamo.registry_abc import RegistryABC
from lalamo.safetensors import safe_read
from lalamo.utils import MapDictValues


def load_torch_weights(path: Path, float_dtype: DTypeLike, *, weights_only: bool = True) -> WeightShard:
    import torch

    from lalamo.modules.torch_interop import torch_to_jax

    torch_weights = torch.load(path, map_location="cpu", weights_only=weights_only)
    return MapDictValues(lambda v: cast_if_float(torch_to_jax(v), float_dtype), torch_weights), {}


class WeightFormat(Enum):
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


# TODO(knyazer): move configs into origin
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
        precision: DTypeLike,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShard]: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @classmethod
    def from_cli(cls, origin_type: str, kwargs: dict[str, str]) -> "Origin":
        name_to_type = {t.__name__: t for t in cls.__descendants__()}
        concrete_type = name_to_type.get(origin_type)
        if concrete_type is None:
            raise ValueError(f"Unknown origin type: {origin_type!r}. Available: {', '.join(sorted(name_to_type))}")
        return concrete_type(**kwargs)


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
    weight_format: WeightFormat = WeightFormat.SAFETENSORS

    def resolve_file(
        self, file_spec: FileSpec, progress_callback: Callable[[StatusEvent], None] | None = None
    ) -> Path:
        return hf_resolve_file(self.repo, file_spec, progress_callback)

    def get_weights(
        self,
        precision: DTypeLike,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShard]:
        paths = hf_resolve_weights(self.repo, self.weight_format.value, progress_callback)
        match self.weight_format:
            case WeightFormat.SAFETENSORS:
                return [safe_read(path, precision) for path in paths]
            case WeightFormat.TORCH:
                return [load_torch_weights(path, precision) for path in paths]

    @property
    def description(self) -> str:
        return self.repo


@dataclass(frozen=True)
class NemoOrigin(Origin):
    repo: str

    def resolve_file(
        self, file_spec: FileSpec, progress_callback: Callable[[StatusEvent], None] | None = None
    ) -> Path:
        if file_spec.filename.endswith(".nemo"):
            (nemo_path,) = hf_resolve_weights(self.repo, ".nemo", progress_callback)
            _, config_path = NemoOrigin.extract_nemo_archive(nemo_path)
            return config_path
        return hf_resolve_file(self.repo, file_spec, progress_callback)

    def get_weights(
        self,
        precision: DTypeLike,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShard]:
        (nemo_path,) = hf_resolve_weights(self.repo, ".nemo", progress_callback)
        weight_paths, _ = NemoOrigin.extract_nemo_archive(nemo_path)
        return [load_torch_weights(path, precision) for path in weight_paths]

    @property
    def description(self) -> str:
        return self.repo

    @classmethod
    @functools.cache
    def extract_nemo_archive(cls, nemo_path: Path) -> tuple[tuple[Path, ...], Path]:
        tmpdir = Path(tempfile.mkdtemp())
        with tarfile.open(nemo_path, "r") as tar:
            for member in tar.getmembers():
                if not (member.name.startswith("..") or Path(member.name).is_absolute() or member.size == 0):
                    tar.extract(member.name, path=tmpdir)

        weights_paths = tuple(tmpdir.glob("*.ckpt"))
        if not weights_paths:
            raise FileNotFoundError("Failed to find Nemo model weights")
        (yaml_path,) = tmpdir.glob("*.yaml")

        with open(yaml_path) as f:
            config_yaml = yaml.safe_load(f)
        config_json_path = yaml_path.with_suffix(".json")
        with open(config_json_path, "w") as f:
            json.dump(config_yaml, f)

        return weights_paths, config_json_path


@dataclass(frozen=True)
class LocalOrigin(Origin):
    root: str
    weight_files: tuple[str, ...] = ()

    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,  # noqa: ARG002
    ) -> Path:
        return Path(self.root) / file_spec.filename

    def get_weights(
        self,
        precision: DTypeLike,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShard]:
        raise NotImplementedError(f"{type(self).__name__} does not support loading weights.")

    @property
    def description(self) -> str:
        return self.root
