from dataclasses import dataclass


@dataclass
class RemoteFileSpec:
    """Specification for a remote file to download."""
    name: str
    url: str
    size: int
    crc32c: str


@dataclass
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
