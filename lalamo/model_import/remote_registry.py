from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteFileSpec:
    """Specification for a remote file to download."""
    name: str
    url: str
    size: int
    crc32c: str


@dataclass(frozen=True)
class RemoteModelSpec:
    """Specification for a pre-converted model in the remote registry."""
    id: str
    vendor: str
    name: str
    family: str
    size: str
    repo_id: str
    quantization: str | None
    files: list[RemoteFileSpec]
