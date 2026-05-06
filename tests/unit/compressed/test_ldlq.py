import jax
import jax.numpy as jnp
import pytest
from einops import rearrange
from jax.lax import DotAlgorithmPreset

from lalamo.compressed.utils.ldlq import block_ldl
from lalamo.utils.precision import use_dot_algorithm_preset
from tests.common import assert_close


def _positive_definite_matrix(size: int) -> jax.Array:
    values = (jnp.arange(size * size, dtype=jnp.float32).reshape(size, size) - 11) / size
    return values @ values.T + jnp.identity(size, dtype=values.dtype)


def _reconstruct_from_block_ldl(matrix: jax.Array, block_size: int) -> jax.Array:
    factors = block_ldl(matrix, block_size)
    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        return factors.tril @ factors.diag @ factors.tril.T


@pytest.mark.parametrize("block_size", [1, 2, 3, 6])
def test_block_ldl_reconstructs_positive_definite_matrix(block_size: int) -> None:
    matrix = _positive_definite_matrix(6)

    result = _reconstruct_from_block_ldl(matrix, block_size)

    assert_close(result=result, reference=matrix, atol=1e-4, rtol=1e-4)


def test_block_ldl_returns_unit_lower_triangular_factor_and_block_diagonal_factor() -> None:
    matrix = _positive_definite_matrix(6)
    block_size = 2

    factors = block_ldl(matrix, block_size)

    assert_close(result=jnp.triu(factors.tril, k=1), reference=jnp.zeros_like(factors.tril))
    assert_close(result=jnp.diag(factors.tril), reference=jnp.ones(6, dtype=matrix.dtype))

    diag_blocks = rearrange(
        factors.diag,
        "(row_blocks block_rows) (col_blocks block_cols) -> row_blocks col_blocks block_rows block_cols",
        block_rows=block_size,
        block_cols=block_size,
    )
    offdiag_rows, offdiag_cols = jnp.tril_indices(3, k=-1)
    offdiag_blocks = diag_blocks[offdiag_rows, offdiag_cols, :, :]
    assert_close(result=offdiag_blocks, reference=jnp.zeros_like(offdiag_blocks))
    assert_close(result=diag_blocks[offdiag_cols, offdiag_rows, :, :], reference=jnp.zeros_like(offdiag_blocks))


def test_block_ldl_matches_cholesky_for_full_matrix_block() -> None:
    matrix = _positive_definite_matrix(4)

    factors = block_ldl(matrix, block_size=4)

    assert_close(result=factors.tril, reference=jnp.identity(4, dtype=matrix.dtype))
    assert_close(result=factors.diag, reference=matrix, atol=1e-4, rtol=1e-4)


def test_block_ldl_can_be_jitted_with_static_block_size() -> None:
    matrix = _positive_definite_matrix(6)
    jitted_block_ldl = jax.jit(block_ldl, static_argnames=("block_size",))

    factors = jitted_block_ldl(matrix, block_size=2)
    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        result = factors.tril @ factors.diag @ factors.tril.T

    assert_close(result=result, reference=matrix, atol=1e-4, rtol=1e-4)


def test_block_ldl_rejects_non_square_matrix() -> None:
    matrix = jnp.ones((2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="must be square"):
        block_ldl(matrix, block_size=1)


def test_block_ldl_rejects_block_size_that_does_not_divide_matrix_size() -> None:
    matrix = _positive_definite_matrix(5)

    with pytest.raises(ValueError, match="must be divisible"):
        block_ldl(matrix, block_size=2)
