from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class RemoteFileSpec:
    name: str
    url: str
    size: int
    crc32c: str


@dataclass(frozen=True)
class RemoteModelSpec:
    id: str
    vendor: str
    name: str
    family: str
    size: str
    repo_id: str
    quantization: str | None
    files: list[RemoteFileSpec]


def fetch_available_models() -> list[RemoteModelSpec]:
    api_url = "https://sdk.trymirai.com/api/v1/models/list/lalamo"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    data = response.json()
    models = []

    for model_data in data.get("models", []):
        files = [
            RemoteFileSpec(
                name=f["name"],
                url=f["url"],
                size=f["size"],
                crc32c=f["crc32c"],
            )
            for f in model_data.get("files", [])
        ]

        model = RemoteModelSpec(
            id=model_data["id"],
            vendor=model_data["vendor"],
            name=model_data["name"],
            family=model_data["family"],
            size=model_data["size"],
            repo_id=model_data["repoId"],
            quantization=model_data.get("quantization"),
            files=files,
        )
        models.append(model)

    return models
