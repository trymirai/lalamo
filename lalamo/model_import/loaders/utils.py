# MXFP4 decoding utilities for model loaders.
# Based on OpenAI's reference implementation logic for GPT-OSS MXFP4 weights.
# Converts packed FP4 blocks plus per-row scales into dense weights in the target dtype.

from collections.abc import Mapping

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float

from lalamo.utils.parameter_path import ParameterPath

__all__ = [
    "decode_mxfp4",
    "deinterleave_pairwise_columns",
    "dequantize_experts",
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


def dequantize_experts(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    experts_per_chunk: int = 8,
) -> Float[Array, "experts rows columns"]:
    weights = weights_dict[path]
    scale_path = ParameterPath(f"{path}_scale")
    amax_path = ParameterPath(f"{path}_amax")
    if amax_path in weights_dict:
        e2m1_values = jnp.asarray(_MXFP4_LUT_VALUES, dtype=jnp.float32)
        block_scales = weights_dict[scale_path]
        denominator_path = ParameterPath(f"{path}_outer_scale_denominator")
        expert_factors = weights_dict[amax_path].astype(jnp.float32) / weights_dict[denominator_path].astype(
            jnp.float32
        )

        def dequantize_chunk(start: int) -> Float[Array, "chunk rows columns"]:
            codes = weights[start : start + experts_per_chunk].astype(jnp.int32)
            values = rearrange(
                jnp.stack([e2m1_values[codes & 0xF], e2m1_values[codes >> 4]], axis=-1),
                "experts rows (groups bytes) nibbles -> experts rows groups (bytes nibbles)",
                bytes=8,
            )
            scales = block_scales[start : start + experts_per_chunk].astype(jnp.float32)[..., None]
            factors = expert_factors[start : start + experts_per_chunk, None, None, None]
            return rearrange(values * scales * factors, "experts rows groups values -> experts rows (groups values)")

        num_experts, _, _ = weights.shape
        return jnp.concatenate([dequantize_chunk(start) for start in range(0, num_experts, experts_per_chunk)])
    if weights.dtype == jnp.float8_e4m3fn:
        values = rearrange(
            weights.astype(jnp.float32),
            "experts rows (groups values) -> experts rows groups values",
            values=32,
        )
        scales = jnp.exp2(weights_dict[scale_path].astype(jnp.float32) - 127)[..., None]
        return rearrange(values * scales, "experts rows groups values -> experts rows (groups values)")
    return weights.astype(jnp.float32)
