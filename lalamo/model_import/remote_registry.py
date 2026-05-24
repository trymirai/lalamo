import importlib.metadata
from dataclasses import dataclass, replace
from typing import ClassVar

import cattrs
import requests


@dataclass(frozen=True)
class RegistryMetadata:
    id: str
    name: str


@dataclass(frozen=True)
class RegistryModelFileHash:
    value: str
    method: str


@dataclass(frozen=True)
class RegistryModelFile:
    url: str
    name: str
    size: int
    hashes: tuple[RegistryModelFileHash, ...]

    @property
    def crc32c(self) -> str:
        return next((file_hash.value for file_hash in self.hashes if file_hash.method == "crc32c"), "")


@dataclass(frozen=True)
class RegistryModelRepository:
    identifier: str


@dataclass(frozen=True)
class RegistryModelReference:
    type: str
    files: tuple[RegistryModelFile, ...]
    repository: RegistryModelRepository
    source_repository: RegistryModelRepository
    toolchain_version: str


@dataclass(frozen=True)
class RegistryModelAccessibility:
    type: str
    reference: RegistryModelReference


@dataclass(frozen=True)
class RegistryModelBackend:
    id: str
    version: str
    metadata_id: str


@dataclass(frozen=True)
class RegistryModelVendor:
    id: str
    metadata_id: str


@dataclass(frozen=True)
class RegistryModelFamily:
    id: str
    metadata_id: str
    vendor: RegistryModelVendor


@dataclass(frozen=True)
class RegistryModelProperties:
    id: str
    metadata_id: str
    size: int | None = None
    version: str | None = None


@dataclass(frozen=True)
class RegistryModelQuantization:
    id: str
    metadata_id: str
    method: str
    bits_per_weight: int | None = None
    vendor: RegistryModelVendor | None = None


@dataclass(frozen=True)
class RegistryModel:
    id: str
    metadata_id: str
    backends: tuple[RegistryModelBackend, ...]
    family: RegistryModelFamily
    properties: RegistryModelProperties
    accessibility: RegistryModelAccessibility
    quantization: RegistryModelQuantization | None = None
    metadatas: tuple[RegistryMetadata, ...] = ()

    def _metadata_name(self, metadata_id: str, fallback: str) -> str:
        return next((metadata.name for metadata in self.metadatas if metadata.id == metadata_id), fallback)

    @property
    def name(self) -> str:
        return self._metadata_name(self.metadata_id, self.repo_id)

    @property
    def vendor(self) -> str:
        return self._metadata_name(self.family.vendor.metadata_id, self.family.vendor.id)

    @property
    def repo_id(self) -> str:
        return self.accessibility.reference.source_repository.identifier

    @property
    def artifact_repo_id(self) -> str:
        return self.accessibility.reference.repository.identifier

    @property
    def files(self) -> tuple[RegistryModelFile, ...]:
        return self.accessibility.reference.files


@dataclass(frozen=True)
class RegistryResponse:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    metadatas: tuple[RegistryMetadata, ...]
    models: tuple[RegistryModel, ...]

    @classmethod
    def from_json(cls, data: object) -> "RegistryResponse":
        return cls._converter.structure(data, cls)


def fetch_available_models() -> list[RegistryModel]:
    response = requests.post(
        "https://sdk.trymirai.com/api/v1/fetch/models",
        json={
            "include_traces": True,
            "backends": [
                {
                    "identifier": "uzu",
                    "version": importlib.metadata.version("lalamo"),
                },
            ],
        },
        timeout=30,
    )
    response.raise_for_status()

    registry_response = RegistryResponse.from_json(response.json())
    return [replace(model, metadatas=registry_response.metadatas) for model in registry_response.models]
