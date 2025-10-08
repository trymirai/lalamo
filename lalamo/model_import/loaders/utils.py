# MXFP4 decoding utilities for model loaders.
# Based on OpenAI's reference implementation logic for GPT-OSS MXFP4 weights.
# Converts packed FP4 blocks plus per-row scales into dense weights in the target dtype.

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

__all__ = [
    "decode_mxfp4",
    "deinterleave_pairwise_columns",
]


# The 16 representable FP4 values used by MXFP4, in logical order (low nibble indices 0..15).
# See: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
_MXFP4_LUT_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def decode_mxfp4(
    blocks: Array,
    scales: Array,
    *,
    dtype: DTypeLike,
    flatten: bool = False,
) -> Array:
    target_dtype = jnp.dtype(dtype)

    # Prepare LUT in target dtype
    lut = jnp.array(_MXFP4_LUT_VALUES, dtype=target_dtype)

    *prefix, rows, packed_cols = blocks.shape
    if scales.shape != (*prefix, rows):
        raise ValueError(
            f"MXFP4 scales shape {scales.shape} does not match blocks prefix/rows {(*prefix, rows)}",
        )

    # Extract low/high nibble indices
    low_mask = jnp.array(0x0F, dtype=blocks.dtype)
    idx_lo = (blocks & low_mask).astype(jnp.int32)
    idx_hi = (blocks >> jnp.array(4, dtype=blocks.dtype)).astype(jnp.int32)

    # Lookup FP4 base values
    vals_lo = lut[idx_lo]
    vals_hi = lut[idx_hi]

    # Interleave into (..., rows, 2*packed_cols)
    out_shape = (*prefix, rows, packed_cols * 2)
    out = jnp.empty(out_shape, dtype=target_dtype)
    out = out.at[..., 0::2].set(vals_lo)
    out = out.at[..., 1::2].set(vals_hi)

    # Apply exponent scaling: exponents are biased by 127 in checkpoints
    exp = scales.astype(jnp.int32) - 127
    out = jnp.ldexp(out, exp[..., None])

    if flatten:
        return out.reshape(*prefix, rows * (packed_cols * 2))
    return out


def deinterleave_pairwise_columns(
    matrix: Array,
    *,
    first: str = "even",
) -> tuple[Array, Array]:
    if matrix.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {matrix.shape[-1]}")

    match first:
        case "even":
            return matrix[..., 0::2], matrix[..., 1::2]
        case "odd":
            return matrix[..., 1::2], matrix[..., 0::2]
        case _:
            raise ValueError("Parameter 'first' must be either 'even' or 'odd'")
