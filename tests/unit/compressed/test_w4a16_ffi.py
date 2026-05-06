from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.lax import DotAlgorithmPreset

from lalamo.compressed import w4a16_ffi

FfiCall = tuple[str, jax.ShapeDtypeStruct, str]


@pytest.mark.parametrize(
    ("dtype", "target"),
    [
        (jnp.float16, "lalamo_w4a16_mlx_f16"),
        (jnp.bfloat16, "lalamo_w4a16_mlx_bf16"),
    ],
)
def test_mlx_w4a16_dot_builds_matvec_ffi_call(monkeypatch: pytest.MonkeyPatch, dtype: jnp.dtype, target: str) -> None:
    calls: list[FfiCall] = []

    def ffi_call(
        target_name: str,
        result_shape_dtype: jax.ShapeDtypeStruct,
        *,
        vmap_method: str,
    ) -> Callable[..., jax.Array]:
        calls.append((target_name, result_shape_dtype, vmap_method))

        def output(*_args: object) -> jax.Array:
            return jnp.zeros(result_shape_dtype.shape, result_shape_dtype.dtype)

        return output

    monkeypatch.setattr(jax.ffi, "ffi_call", ffi_call)

    result = w4a16_ffi.mlx_w4a16_dot(
        jnp.ones((64,), dtype=dtype),
        jnp.ones((8, 32), dtype=jnp.uint8),
        jnp.ones((8, 2), dtype=dtype),
        jnp.ones((8, 2), dtype=dtype),
    )

    target_name, result_shape_dtype, vmap_method = calls.pop()
    assert target_name == target
    assert result_shape_dtype.shape == (8,)
    assert result_shape_dtype.dtype == dtype
    assert vmap_method == "sequential"
    assert result.shape == (8,)


@pytest.mark.parametrize(
    ("dtype", "target"),
    [
        (jnp.float16, "lalamo_w4a16_awq_f16"),
        (jnp.bfloat16, "lalamo_w4a16_awq_bf16"),
    ],
)
def test_awq_w4a16_dot_builds_batched_ffi_call(monkeypatch: pytest.MonkeyPatch, dtype: jnp.dtype, target: str) -> None:
    calls: list[FfiCall] = []

    def ffi_call(
        target_name: str,
        result_shape_dtype: jax.ShapeDtypeStruct,
        *,
        vmap_method: str,
    ) -> Callable[..., jax.Array]:
        calls.append((target_name, result_shape_dtype, vmap_method))

        def output(*_args: object) -> jax.Array:
            return jnp.zeros(result_shape_dtype.shape, result_shape_dtype.dtype)

        return output

    monkeypatch.setattr(jax.ffi, "ffi_call", ffi_call)

    result = w4a16_ffi.awq_w4a16_dot(
        jnp.ones((4, 64), dtype=dtype),
        jnp.ones((8, 32), dtype=jnp.uint8),
        jnp.ones((8, 2), dtype=dtype),
        jnp.ones((8, 1), dtype=jnp.uint8),
    )

    target_name, result_shape_dtype, vmap_method = calls.pop()
    assert target_name == target
    assert result_shape_dtype.shape == (4, 8)
    assert result_shape_dtype.dtype == dtype
    assert vmap_method == "sequential"
    assert result.shape == (4, 8)


def test_w4a16_dot_rejects_unsupported_shape() -> None:
    with pytest.raises(ValueError, match="group size 32"):
        w4a16_ffi.mlx_w4a16_dot(
            jnp.ones((64,), dtype=jnp.float16),
            jnp.ones((8, 32), dtype=jnp.uint8),
            jnp.ones((8, 1), dtype=jnp.float16),
            jnp.ones((8, 1), dtype=jnp.float16),
        )


def test_w4a16_dot_rejects_non_default_precision() -> None:
    with pytest.raises(ValueError, match="only support DEFAULT"):
        w4a16_ffi.mlx_w4a16_dot(
            jnp.ones((64,), dtype=jnp.float16),
            jnp.ones((8, 32), dtype=jnp.uint8),
            jnp.ones((8, 2), dtype=jnp.float16),
            jnp.ones((8, 2), dtype=jnp.float16),
            precision=DotAlgorithmPreset.F32_F32_F32,
        )
