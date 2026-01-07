import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from io import BufferedReader, BufferedWriter
from typing import Any, ClassVar, Self

import cattrs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from lalamo.utils import LazyDict

SF2J = {
    "BOOL": jnp.dtype(jnp.bool_),
    "U8": jnp.dtype(jnp.uint8),
    "I8": jnp.dtype(jnp.int8),
    "I16": jnp.dtype(jnp.int16),
    "U16": jnp.dtype(jnp.uint16),
    "F16": jnp.dtype(jnp.float16),
    "BF16": jnp.dtype(jnp.bfloat16),
    "I32": jnp.dtype(jnp.int32),
    "U32": jnp.dtype(jnp.uint32),
    "F32": jnp.dtype(jnp.float32),
    "C64": jnp.dtype(jnp.complex64),
    "F64": jnp.dtype(jnp.float64),
    "I64": jnp.dtype(jnp.int64),
    "U64": jnp.dtype(jnp.uint64),
}

J2SF = {v: k for k, v in SF2J.items()}


@dataclass(frozen=True)
class SFTensorInfo:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(jnp.dtype, lambda x, _: SF2J[x])
    _converter.register_unstructure_hook(jnp.dtype, lambda x: J2SF[x])

    dtype: jnp.dtype
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def start(self) -> int:
        return self.data_offsets[0]

    @property
    def end(self) -> int:
        return self.data_offsets[1]

    @property
    def size(self) -> int:
        return self.end - self.start

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls._converter.structure(obj, cls)

    def to_dict(self) -> dict:
        return self._converter.unstructure(self)


def safe_read(fd: BufferedReader) -> tuple[dict[str, str] | None, LazyDict[str, Array]]:
    header_size = struct.unpack("<Q", fd.read(8))[0]
    header: dict[str, dict[str, Any]] = json.loads(fd.read(header_size))
    metadata: dict[str, str] | None = header.pop("__metadata__", None)
    data_offset = fd.tell()

    def _load_tensor(key: str) -> Array:
        info = SFTensorInfo.from_dict(header[key])
        fd.seek(data_offset + info.start)
        return jnp.asarray(np.fromfile(fd, info.dtype, info.size // info.dtype.itemsize)).reshape(info.shape)

    lazy_tensors = LazyDict(set(header.keys()), _load_tensor)
    return (metadata, lazy_tensors)


def safe_write(fd: BufferedWriter, tensors: Mapping[str, Array]) -> None:
    sorted_tensors = dict(sorted(tensors.items(), key=lambda x: (-x[1].dtype.alignment, x[0])))

    header_dict = {}
    offset = 0
    for key, tensor in sorted_tensors.items():
        assert offset % tensor.dtype.alignment == 0
        header_dict[key] = SFTensorInfo(tensor.dtype, tensor.shape, (offset, offset + tensor.nbytes)).to_dict()
        offset += tensor.nbytes

    data_alignment = max(8, next((t.dtype.alignment for t in sorted_tensors.values()), 1))
    header = json.dumps(header_dict).encode()
    header += b" " * (-len(header) % data_alignment)
    fd.write(struct.pack("<Q", len(header)) + header)

    for tensor in sorted_tensors.values():
        jax.device_get(tensor).tofile(fd)
