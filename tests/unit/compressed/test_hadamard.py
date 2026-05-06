from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed.utils.hadamard import hadamard_transform
from tests.common import assert_close, gpu_only, tolerance


def _sylvester_hadamard_matrix(block_size: int) -> jax.Array:
    result = jnp.ones((1, 1), dtype=jnp.float32)
    for _ in range(block_size.bit_length() - 1):
        result = jnp.block(
            [
                [result, result],
                [result, -result],
            ],
        )
    return result


def _dense_hadamard_transform(inputs: jax.Array, block_size: int) -> jax.Array:
    (input_dim,) = inputs.shape
    transform_matrix = _sylvester_hadamard_matrix(block_size).astype(inputs.dtype)
    transform_matrix = transform_matrix / jnp.sqrt(jnp.asarray(block_size, dtype=inputs.dtype))
    blocks = jnp.reshape(inputs, (input_dim // block_size, block_size))
    with jax.default_matmul_precision("float32"):
        transformed_blocks = jax.vmap(lambda block: transform_matrix @ block)(blocks)
    return jnp.reshape(transformed_blocks, (input_dim,))


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_hadamard_transform_matches_dense_reference(block_size: Literal[32, 64, 128]) -> None:
    inputs = (jnp.arange(2 * block_size, dtype=jnp.float32) - 3) / 5

    result = hadamard_transform(inputs, block_size)

    assert result.dtype == inputs.dtype
    _assert_close(result=result, reference=_dense_hadamard_transform(inputs, block_size))


def test_hadamard_transform_is_its_own_inverse() -> None:
    inputs = (jnp.arange(128, dtype=jnp.float32) - 7) / 3

    result = hadamard_transform(hadamard_transform(inputs, block_size=64), block_size=64)

    _assert_close(result=result, reference=inputs)


def test_hadamard_transform_preserves_bfloat16_dtype() -> None:
    inputs = jnp.arange(64, dtype=jnp.bfloat16)

    result = hadamard_transform(inputs, block_size=32)

    assert result.dtype == inputs.dtype
    _assert_close(
        result=result.astype(jnp.float32), reference=_dense_hadamard_transform(inputs, 32).astype(jnp.float32)
    )


@pytest.mark.parametrize("block_size", [0, 8, 16, 256])
def test_hadamard_transform_rejects_unsupported_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="one of 32, 64, or 128"):
        hadamard_transform(jnp.ones((256,), dtype=jnp.float32), block_size)  # type: ignore[arg-type]


def test_hadamard_transform_rejects_input_dimension_not_divisible_by_block_size() -> None:
    with pytest.raises(ValueError, match="multiple of block size"):
        hadamard_transform(jnp.ones((96,), dtype=jnp.float32), block_size=64)


def test_hadamard_transform_jit_matches_dense_reference() -> None:
    inputs = (jnp.arange(128, dtype=jnp.float32) - 3) / 2

    result = jax.jit(lambda inputs: hadamard_transform(inputs, block_size=32))(inputs)

    _assert_close(result=result, reference=_dense_hadamard_transform(inputs, block_size=32))


def test_hadamard_transform_vmap_matches_dense_reference() -> None:
    inputs = (jnp.reshape(jnp.arange(3 * 128, dtype=jnp.float32), (3, 128)) - 5) / 7

    result = jax.vmap(lambda inputs: hadamard_transform(inputs, block_size=64))(inputs)
    reference = jax.vmap(lambda inputs: _dense_hadamard_transform(inputs, block_size=64))(inputs)

    with tolerance(atol=2e-3, rtol=3e-2):
        _assert_close(result=result, reference=reference)


def test_hadamard_transform_custom_vjp_matches_self_adjoint_reference() -> None:
    inputs = (jnp.arange(64, dtype=jnp.float32) - 2) / 3
    cotangent = (jnp.arange(64, dtype=jnp.float32) + 1) / 7

    result = jax.grad(
        lambda inputs: jnp.vdot(hadamard_transform(inputs, block_size=32), cotangent),
    )(inputs)

    _assert_close(result=result, reference=hadamard_transform(cotangent, block_size=32))


def test_hadamard_transform_vjp_matches_self_adjoint_reference() -> None:
    inputs = (jnp.arange(128, dtype=jnp.float32) - 4) / 5
    cotangent = (jnp.arange(128, dtype=jnp.float32) + 3) / 11

    _, vjp_fn = jax.vjp(lambda inputs: hadamard_transform(inputs, block_size=64), inputs)
    (result,) = vjp_fn(cotangent)

    _assert_close(result=result, reference=hadamard_transform(cotangent, block_size=64))


@gpu_only
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_hadamard_transform_cute_matches_dense_reference_on_gpu(
    block_size: Literal[32, 64, 128],
    dtype: jnp.dtype,
) -> None:
    inputs = ((jnp.arange(block_size, dtype=jnp.float32) % 23) / 11).astype(dtype)

    result = jax.jit(lambda inputs: hadamard_transform(inputs, block_size=block_size))(inputs)

    with tolerance(atol=2e-2, rtol=3e-2):
        _assert_close(result=result, reference=_dense_hadamard_transform(inputs, block_size=block_size))


@gpu_only
def test_hadamard_transform_cute_custom_vjp_matches_self_adjoint_reference_on_gpu() -> None:
    inputs = (jnp.arange(128, dtype=jnp.float32) - 7) / 13
    cotangent = jnp.sin(jnp.arange(128, dtype=jnp.float32))

    _, vjp_fn = jax.vjp(lambda inputs: hadamard_transform(inputs, block_size=128), inputs)
    (result,) = vjp_fn(cotangent)

    _assert_close(result=result, reference=hadamard_transform(cotangent, block_size=128))


@gpu_only
def test_hadamard_transform_cute_vmap_matches_dense_reference_on_gpu() -> None:
    inputs = ((jnp.reshape(jnp.arange(5 * 64 * 4, dtype=jnp.float32), (5, 64 * 4)) % 29) / 17).astype(jnp.bfloat16)

    result = jax.jit(jax.vmap(lambda inputs: hadamard_transform(inputs, block_size=64)))(inputs)
    reference = jax.vmap(lambda inputs: _dense_hadamard_transform(inputs, block_size=64))(inputs)

    with tolerance(atol=3e-2, rtol=3e-2):
        _assert_close(result=result, reference=reference)
