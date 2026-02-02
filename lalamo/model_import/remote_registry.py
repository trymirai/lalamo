from dataclasses import dataclass
from typing import Any, ClassVar

import cattrs
import requests


@dataclass(frozen=True)
class RegistryModelFile:
    name: str
    url: str
    size: int
    crc32c: str


@dataclass(frozen=True)
class RegistryModel:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    id: str
    vendor: str
    name: str
    family: str
    size: str
    repo_id: str
    quantization: str | None
    files: list[RegistryModelFile]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegistryModel":
        if "repoId" in data:
            data = {**data, "repo_id": data.pop("repoId")}
        return cls._converter.structure(data, cls)


def fetch_available_models() -> list[RegistryModel]:
    api_url = "https://sdk.trymirai.com/api/v1/models/list/lalamo"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    data = response.json()
    models_data = data.get("models", [])

    return [RegistryModel.from_dict(model_data) for model_data in models_data]
