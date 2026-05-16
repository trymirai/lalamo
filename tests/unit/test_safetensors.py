from pathlib import Path

import jax.numpy as jnp

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
