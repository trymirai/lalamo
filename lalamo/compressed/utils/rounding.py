from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.module import Keychain
from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays
from lalamo.weight_matrix import GradientEstimator

from .grouping import unsigned_qmax

__all__ = [
    "deterministic_round_to_unsigned_grid",
    "round_to_unsigned_grid",
    "round_up_to_e4m3",
    "stochastic_round_to_unsigned_grid",
]

_E4M3_LEVELS = (
    *(mantissa / 512 for mantissa in range(1, 8)),
    *(2 ** (exponent - 7) * (1 + mantissa / 8) for exponent in range(1, 15) for mantissa in range(8)),
    *(2 ** (15 - 7) * (1 + mantissa / 8) for mantissa in range(7)),
)


def _mask_straight_through_gradients(
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


def _deterministic_round_to_unsigned_grid_impl(values: Float[Array, "..."], *, bits: int) -> Float[Array, "..."]:
    return jnp.round(jnp.clip(values, 0, unsigned_qmax(bits)))


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _deterministic_round_to_unsigned_grid(values: Float[Array, "..."], bits: int) -> Float[Array, "..."]:
    return _deterministic_round_to_unsigned_grid_impl(values, bits=bits)


def _deterministic_round_to_unsigned_grid_fwd(
    values: Float[Array, "..."],
    bits: int,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    rounded_values = _deterministic_round_to_unsigned_grid_impl(values, bits=bits)
    return rounded_values, values


def _deterministic_round_to_unsigned_grid_bwd(
    bits: int,
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
) -> tuple[Float[Array, "..."]]:
    return (_mask_straight_through_gradients(values, gradients, bits=bits),)


_deterministic_round_to_unsigned_grid.defvjp(
    _deterministic_round_to_unsigned_grid_fwd,
    _deterministic_round_to_unsigned_grid_bwd,
)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def deterministic_round_to_unsigned_grid(
    values: Float[Array, "..."],
    *,
    bits: int,
) -> Float[Array, "..."]:
    return _deterministic_round_to_unsigned_grid(values, bits)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def round_up_to_e4m3(values: Float[Array, "..."]) -> Float[Array, "..."]:
    levels = jnp.array(_E4M3_LEVELS, dtype=jnp.float32)
    clipped_values = jnp.maximum(values.astype(jnp.float32), jnp.finfo(jnp.float32).tiny)
    indices = jnp.minimum(jnp.sum(clipped_values[..., None] > levels, axis=-1), levels.size - 1)
    return jnp.sum((indices[..., None] == jnp.arange(levels.size)).astype(jnp.float32) * levels, axis=-1)


def _stochastic_round_to_unsigned_grid_impl(
    values: Float[Array, "..."],
    *,
    bits: int,
    keychain: Keychain,
) -> Float[Array, "..."]:
    clipped_values = jnp.clip(values, 0, unsigned_qmax(bits))
    lower_bins = jnp.floor(clipped_values)
    upper_probability = clipped_values - lower_bins
    upper_samples = (
        jax.random.uniform(keychain.batch_key, clipped_values.shape, dtype=clipped_values.dtype) < upper_probability
    )
    return lower_bins + upper_samples.astype(clipped_values.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _stochastic_round_to_unsigned_grid(
    values: Float[Array, "..."],
    bits: int,
    rounding_keychain: Keychain,
) -> Float[Array, "..."]:
    return _stochastic_round_to_unsigned_grid_impl(values, bits=bits, keychain=rounding_keychain)


def _stochastic_round_to_unsigned_grid_fwd(
    values: Float[Array, "..."],
    bits: int,
    rounding_keychain: Keychain,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    rounded_values = _stochastic_round_to_unsigned_grid_impl(values, bits=bits, keychain=rounding_keychain)
    return rounded_values, values


def _stochastic_round_to_unsigned_grid_bwd(
    bits: int,
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
) -> tuple[Float[Array, "..."], None]:
    return _mask_straight_through_gradients(values, gradients, bits=bits), None


_stochastic_round_to_unsigned_grid.defvjp(
    _stochastic_round_to_unsigned_grid_fwd,
    _stochastic_round_to_unsigned_grid_bwd,
)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def stochastic_round_to_unsigned_grid(
    values: Float[Array, "..."],
    *,
    bits: int,
    keychain: Keychain,
) -> Float[Array, "..."]:
    return _stochastic_round_to_unsigned_grid(values, bits, keychain)


def round_to_unsigned_grid(
    values: Float[Array, "..."],
    *,
    bits: int,
    keychain: Keychain,
    gradient_estimator: GradientEstimator,
) -> Float[Array, "..."]:
    if gradient_estimator == GradientEstimator.DETERMINISTIC_ROUNDING:
        return deterministic_round_to_unsigned_grid(values, bits=bits)
    if gradient_estimator == GradientEstimator.STOCHASTIC_ROUNDING:
        return stochastic_round_to_unsigned_grid(values, bits=bits, keychain=keychain)
    if gradient_estimator == GradientEstimator.LOCAL_ADDITIVE_NOISE:
        raise ValueError("Local additive noise is not implemented.")
    raise ValueError(f"Unsupported gradient estimator {gradient_estimator}")
