from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Float

from lalamo.utils.precision import use_dot_algorithm_preset

__all__ = [
    "LDLFactors",
    "block_ldl",
]


class LDLFactors(NamedTuple):
    tril: Float[Array, "n n"]
    diag: Float[Array, "n n"]


def block_ldl(matrix: Float[Array, "n n"], block_size: int) -> LDLFactors:
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    if rows % block_size != 0:
        raise ValueError(f"rows={rows} must be divisible by block_size={block_size}")

    num_blocks = rows // block_size
    cholesky_tril = jnp.linalg.cholesky(matrix)
    cholesky_blocks = rearrange(
        cholesky_tril,
        "(row_blocks block_rows) (col_blocks block_cols) -> row_blocks col_blocks block_rows block_cols",
        row_blocks=num_blocks,
        col_blocks=num_blocks,
    )

    diag_block_indices = jnp.arange(num_blocks)
    cholesky_block_diag = cholesky_blocks[diag_block_indices, diag_block_indices, :, :]

    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        result_diag_blocks = cholesky_block_diag @ jnp.swapaxes(cholesky_block_diag, -1, -2)

    result_diag_buffer = jnp.zeros_like(cholesky_blocks)
    result_diag_buffer = result_diag_buffer.at[diag_block_indices, diag_block_indices, :, :].set(result_diag_blocks)
    result_diag = rearrange(
        result_diag_buffer,
        "row_blocks col_blocks block_rows block_cols -> (row_blocks block_rows) (col_blocks block_cols)",
        row_blocks=num_blocks,
        col_blocks=num_blocks,
    )

    offdiag_rows, offdiag_cols = jnp.tril_indices(num_blocks, k=-1)
    offdiag_blocks = cholesky_blocks[offdiag_rows, offdiag_cols, :, :]
    diag_blocks = cholesky_blocks[offdiag_cols, offdiag_cols, :, :]

    solved_transposed = jax.vmap(
        lambda left_tril_t, rhs_t: jax.scipy.linalg.solve_triangular(
            left_tril_t,
            rhs_t,
            lower=False,
            trans="N",
            unit_diagonal=False,
        ),
    )(jnp.swapaxes(diag_blocks, -1, -2), jnp.swapaxes(offdiag_blocks, -1, -2))
    result_tril_blocks = jnp.swapaxes(solved_transposed, -1, -2)

    result_tril_buffer = rearrange(
        jnp.identity(rows, dtype=matrix.dtype),
        "(row_blocks block_rows) (col_blocks block_cols) -> row_blocks col_blocks block_rows block_cols",
        row_blocks=num_blocks,
        col_blocks=num_blocks,
    )
    result_tril_buffer = result_tril_buffer.at[offdiag_rows, offdiag_cols, :, :].set(result_tril_blocks)
    result_tril = rearrange(
        result_tril_buffer,
        "row_blocks col_blocks block_rows block_cols -> (row_blocks block_rows) (col_blocks block_cols)",
        row_blocks=num_blocks,
        col_blocks=num_blocks,
    )

    return LDLFactors(tril=result_tril, diag=result_diag)
