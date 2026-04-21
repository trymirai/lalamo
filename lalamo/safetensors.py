import json
import mmap
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from io import BufferedWriter
from pathlib import Path
from typing import ClassVar, Self

import cattrs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike

from lalamo.common import WeightShard, cast_if_float
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


# mmap is at least as fast as lazy fd reads, while not requiring an open descriptor.
# See https://arxiv.org/abs/2505.23072v1 for analysis of safetensors I/O patterns.
def safe_read(path: Path | str, float_dtype: DTypeLike | None = None) -> WeightShard:
    path = Path(path)
    fd = path.open("rb")
    mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
    fd.close()

    header_size = struct.unpack("<Q", mm[:8])[0]
    header: dict[str, dict] = json.loads(mm[8 : 8 + header_size])
    metadata: dict[str, str] = header.pop("__metadata__", None) or {}
    data_offset = 8 + header_size

    def _load_tensor(key: str) -> Array:
        info = SFTensorInfo.from_dict(header[key])
        buf = mm[data_offset + info.start : data_offset + info.end]
        tensor = jnp.asarray(np.frombuffer(buf, dtype=info.dtype).reshape(info.shape))
        if float_dtype is not None:
            tensor = cast_if_float(tensor, float_dtype)
        return tensor

    return LazyDict(set(header.keys()), _load_tensor), metadata


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
