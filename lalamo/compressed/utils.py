from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Key

from lalamo.weight_matrix import GradientEstimator, MatmulConfig

__all__ = [
    "MinMax",
    "expand_last_axis_groups",
    "group_by_last_axis",
    "grouped_last_axis_shape",
    "min_max_scale",
    "min_max_within_groups",
    "round_to_unsigned_grid",
    "round_to_unsigned_grid_for_config",
    "stochastic_round_to_unsigned_grid",
    "unsigned_qmax",
]


class MinMax(NamedTuple):
    min: Float[Array, "... out_channels in_channels"]
    max: Float[Array, "... out_channels in_channels"]


def grouped_last_axis_shape(shape: tuple[int, ...], *, group_size: int) -> tuple[int, ...]:
    *leading_dims, last_dim = shape
    if last_dim % group_size != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by group size {group_size}")
    return (*leading_dims, last_dim // group_size)


def group_by_last_axis(
    weights: Float[Array, "... out_channels in_channels"],
    *,
    group_size: int,
) -> Float[Array, "... out_channels groups group_channels"]:
    return rearrange(
        weights,
        "... out_channels (groups group_size) -> ... out_channels groups group_size",
        group_size=group_size,
    )


def expand_last_axis_groups(grouped: Float[Array, "... groups"], *, group_size: int) -> Float[Array, "..."]:
    return jnp.repeat(grouped, group_size, axis=-1)


def min_max_within_groups(weights: Float[Array, "... out_channels groups group_channels"]) -> MinMax:
    return MinMax(
        min=jnp.min(weights, axis=-1),
        max=jnp.max(weights, axis=-1),
    )


def unsigned_qmax(bits: int) -> int:
    return (2**bits) - 1


def min_max_scale(min_max: MinMax, *, bits: int, dtype: DTypeLike) -> Float[Array, "..."]:
    finfo = jnp.finfo(dtype)
    scale_range = min_max.max / unsigned_qmax(bits) - min_max.min / unsigned_qmax(bits)
    scales = jnp.maximum(scale_range, finfo.eps)
    return jnp.nan_to_num(scales, nan=finfo.eps, posinf=finfo.max, neginf=finfo.eps)


def _clip_to_unsigned_grid(values: Float[Array, "..."], *, bits: int) -> Float[Array, "..."]:
    qmax = unsigned_qmax(bits)
    finite_values = jnp.nan_to_num(values, nan=0, posinf=qmax, neginf=0)
    return jnp.clip(finite_values, 0, qmax)


def _round_to_unsigned_grid_impl(values: Float[Array, "..."], *, bits: int) -> Float[Array, "..."]:
    return jnp.round(_clip_to_unsigned_grid(values, bits=bits))


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _round_to_unsigned_grid(values: Float[Array, "..."], bits: int) -> Float[Array, "..."]:
    return _round_to_unsigned_grid_impl(values, bits=bits)


def _round_to_unsigned_grid_fwd(
    values: Float[Array, "..."],
    bits: int,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    rounded_values = _round_to_unsigned_grid_impl(values, bits=bits)
    return rounded_values, values


def _clip_unsigned_grid_gradients(
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
    *,
    bits: int,
) -> Float[Array, "..."]:
    qmax = unsigned_qmax(bits)
    inside_range = (values >= 0) & (values <= qmax)
    below_range_with_allowed_gradient = (values < 0) & (gradients > 0)
    above_range_with_allowed_gradient = (values > qmax) & (gradients < 0)
    allowed_gradients = inside_range | below_range_with_allowed_gradient | above_range_with_allowed_gradient
    return jnp.where(allowed_gradients, gradients, 0)


def _round_to_unsigned_grid_bwd(
    bits: int,
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
) -> tuple[Float[Array, "..."]]:
    return (_clip_unsigned_grid_gradients(values, gradients, bits=bits),)


_round_to_unsigned_grid.defvjp(_round_to_unsigned_grid_fwd, _round_to_unsigned_grid_bwd)


def round_to_unsigned_grid(values: Float[Array, "..."], *, bits: int) -> Float[Array, "..."]:
    return _round_to_unsigned_grid(values, bits)


def _stochastic_round_to_unsigned_grid_impl(
    values: Float[Array, "..."],
    *,
    bits: int,
    dequant_key: Key[Array, ""],
) -> Float[Array, "..."]:
    clipped_values = _clip_to_unsigned_grid(values, bits=bits)
    lower_bins = jnp.floor(clipped_values)
    upper_probability = clipped_values - lower_bins
    upper_samples = (
        jax.random.uniform(dequant_key, clipped_values.shape, dtype=clipped_values.dtype) < upper_probability
    )
    return lower_bins + upper_samples.astype(clipped_values.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _stochastic_round_to_unsigned_grid(
    values: Float[Array, "..."],
    bits: int,
    dequant_key: Key[Array, ""],
) -> Float[Array, "..."]:
    return _stochastic_round_to_unsigned_grid_impl(values, bits=bits, dequant_key=dequant_key)


def _stochastic_round_to_unsigned_grid_fwd(
    values: Float[Array, "..."],
    bits: int,
    dequant_key: Key[Array, ""],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    rounded_values = _stochastic_round_to_unsigned_grid_impl(values, bits=bits, dequant_key=dequant_key)
    return rounded_values, values


def _stochastic_round_to_unsigned_grid_bwd(
    bits: int,
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
) -> tuple[Float[Array, "..."], None]:
    return _clip_unsigned_grid_gradients(values, gradients, bits=bits), None


_stochastic_round_to_unsigned_grid.defvjp(
    _stochastic_round_to_unsigned_grid_fwd,
    _stochastic_round_to_unsigned_grid_bwd,
)


def stochastic_round_to_unsigned_grid(
    values: Float[Array, "..."],
    *,
    bits: int,
    dequant_key: Key[Array, ""],
) -> Float[Array, "..."]:
    return _stochastic_round_to_unsigned_grid(values, bits, dequant_key)


def round_to_unsigned_grid_for_config(
    values: Float[Array, "..."],
    *,
    bits: int,
    dequant_key: Key[Array, ""],
    forward_pass_config: MatmulConfig,
) -> Float[Array, "..."]:
    if forward_pass_config.gradient_estimator == GradientEstimator.DETERMINISTIC_ROUNDING:
        return round_to_unsigned_grid(values, bits=bits)
    if forward_pass_config.gradient_estimator == GradientEstimator.STOCHASTIC_ROUNDING:
        return stochastic_round_to_unsigned_grid(values, bits=bits, dequant_key=dequant_key)
    if forward_pass_config.gradient_estimator == GradientEstimator.LOCAL_ADDITIVE_NOISE:
        raise ValueError("Local additive noise is not implemented.")
    raise ValueError(f"Unsupported gradient estimator {forward_pass_config.gradient_estimator}")
