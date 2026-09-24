import struct
from pathlib import Path

import jax.numpy as jnp
from jax import ShapeDtypeStruct

from lalamo.safetensors import safe_read, safe_write


def test_safe_write_roundtrips_float8_tensors(tmp_path: Path) -> None:
    path = tmp_path / "float8.safetensors"
    tensors = {
        "e4m3": jnp.asarray([1.0, 2.0], dtype=jnp.float8_e4m3fn),
        "e5m2": jnp.asarray([1.0, 2.0], dtype=jnp.float8_e5m2),
        "e8m0": jnp.asarray([1.0, 2.0], dtype=jnp.float8_e8m0fnu),
    }

    with path.open("wb") as fd:
        safe_write(fd, tensors)

    with path.open("rb") as fd:
        metadata, restored_tensors = safe_read(fd)
        restored = {name: restored_tensors[name] for name in tensors}

    assert metadata is None
    for name, tensor in tensors.items():
        restored_tensor = restored[name]
        assert restored_tensor.dtype == tensor.dtype
        assert jnp.array_equal(restored_tensor, tensor)


def test_safe_read_without_weights_reads_shapes_from_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "header_only.safetensors"
    tensors = {"a": jnp.zeros((2, 3), dtype=jnp.bfloat16), "b": jnp.ones((4,), dtype=jnp.int32)}

    with path.open("wb") as fd:
        safe_write(fd, tensors, metadata={"key": "value"})
    with path.open("rb") as fd:
        (header_size,) = struct.unpack("<Q", fd.read(8))
    with path.open("r+b") as fd:
        fd.truncate(8 + header_size)

    with path.open("rb") as fd:
        metadata, restored_tensors = safe_read(fd, empty_weights=True)
        restored = {name: restored_tensors[name] for name in tensors}

    assert metadata == {"key": "value"}
    for name, tensor in tensors.items():
        assert isinstance(restored[name], ShapeDtypeStruct)
        assert restored[name].shape == tensor.shape
        assert restored[name].dtype == tensor.dtype
