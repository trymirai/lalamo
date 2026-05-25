from functools import cache, partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, Float, Int, UInt8

from lalamo.module import Keychain
from lalamo.utils.dummy_array import preserve_first_input_sharding, supports_dummy_arrays
from lalamo.utils.sharding import sharding_of, with_sharding
from lalamo.weight_matrix import GradientEstimator

from .grouping import unsigned_qmax

__all__ = [
    "deterministic_round_to_minifloat",
    "deterministic_round_to_sorted_lut_table",
    "deterministic_round_to_unsigned_grid",
    "e8m0_scale_values",
    "lut_values_at",
    "pack_e4m3_scales",
    "pack_e8m0_scales",
    "round_to_minifloat",
    "round_to_sorted_lut_table",
    "round_to_sorted_lut_table_indices",
    "round_to_unsigned_grid",
    "stochastic_round_to_minifloat",
    "stochastic_round_to_sorted_lut_table",
    "stochastic_round_to_unsigned_grid",
]

_E8M0_BIAS = 127


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


def lut_values_at(
    indices: Int[Array, "..."] | UInt8[Array, "..."],
    table: Float[Array, " levels"],
) -> Float[Array, "..."]:
    index_sharding = sharding_of(indices)
    indices = with_sharding(indices.astype(jnp.int32), index_sharding)
    return table.at[indices].get(out_sharding=index_sharding)


def _deterministic_round_to_sorted_lut_table_indices(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> UInt8[Array, "..."]:
    thresholds = (table[:-1] + table[1:]).astype(values.dtype) / 2
    indices = jnp.searchsorted(thresholds, values, side="left", method="compare_all").astype(jnp.uint8)
    return with_sharding(indices, sharding_of(values))


def _stochastic_round_to_sorted_lut_table_indices(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
    keychain: Keychain,
) -> UInt8[Array, "..."]:
    value_sharding = sharding_of(values)
    table = table.astype(values.dtype)
    upper_indices = jnp.clip(jnp.searchsorted(table, values, side="left", method="compare_all"), 1, table.size - 1)
    lower_indices = upper_indices - 1
    upper_indices = with_sharding(upper_indices, value_sharding)
    lower_indices = with_sharding(lower_indices, value_sharding)
    lower_values = lut_values_at(lower_indices, table)
    upper_values = lut_values_at(upper_indices, table)
    upper_probability = jnp.clip(
        (values - lower_values) / jnp.maximum(upper_values - lower_values, jnp.finfo(values.dtype).eps),
        0,
        1,
    )
    samples = jax.random.uniform(
        keychain.batch_key,
        values.shape,
        dtype=values.dtype,
        out_sharding=value_sharding,
    )
    upper_probability = with_sharding(upper_probability, value_sharding)
    return jnp.where(samples < upper_probability, upper_indices, lower_indices).astype(jnp.uint8)


def round_to_sorted_lut_table_indices(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
    *,
    keychain: Keychain | None = None,
    gradient_estimator: GradientEstimator,
) -> UInt8[Array, "..."]:
    if gradient_estimator == GradientEstimator.DETERMINISTIC_ROUNDING:
        return _deterministic_round_to_sorted_lut_table_indices(values, table)
    if gradient_estimator == GradientEstimator.STOCHASTIC_ROUNDING:
        if keychain is None:
            raise ValueError("Stochastic LUT rounding requires a keychain.")
        return _stochastic_round_to_sorted_lut_table_indices(values, table, keychain)
    if gradient_estimator == GradientEstimator.LOCAL_ADDITIVE_NOISE:
        raise ValueError("Local additive noise is not implemented.")
    raise ValueError(f"Unsupported gradient estimator {gradient_estimator}")


def _mask_sorted_lut_table_gradients(
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> Float[Array, "..."]:
    lower_bound = table[0].astype(values.dtype)
    upper_bound = table[-1].astype(values.dtype)
    inside_range = (values >= lower_bound) & (values <= upper_bound)
    below_range_with_allowed_gradient = (values < lower_bound) & (gradients > 0)
    above_range_with_allowed_gradient = (values > upper_bound) & (gradients < 0)
    allowed_gradients = inside_range | below_range_with_allowed_gradient | above_range_with_allowed_gradient
    return jnp.where(allowed_gradients, gradients, 0)


def _deterministic_round_to_sorted_lut_table_impl(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> Float[Array, "..."]:
    return lut_values_at(
        _deterministic_round_to_sorted_lut_table_indices(values, table),
        table,
    )


@jax.custom_vjp
def _deterministic_round_to_sorted_lut_table(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> Float[Array, "..."]:
    return _deterministic_round_to_sorted_lut_table_impl(values, table)


def _deterministic_round_to_sorted_lut_table_fwd(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> tuple[Float[Array, "..."], tuple[Float[Array, "..."], Float[Array, " levels"]]]:
    rounded_values = _deterministic_round_to_sorted_lut_table_impl(values, table)
    return rounded_values, (values, table)


def _deterministic_round_to_sorted_lut_table_bwd(
    residuals: tuple[Float[Array, "..."], Float[Array, " levels"]],
    gradients: Float[Array, "..."],
) -> tuple[Float[Array, "..."], None]:
    values, table = residuals
    return _mask_sorted_lut_table_gradients(values, gradients, table), None


_deterministic_round_to_sorted_lut_table.defvjp(
    _deterministic_round_to_sorted_lut_table_fwd,
    _deterministic_round_to_sorted_lut_table_bwd,
)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def deterministic_round_to_sorted_lut_table(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> Float[Array, "..."]:
    return _deterministic_round_to_sorted_lut_table(values, table)


def _stochastic_round_to_sorted_lut_table_impl(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
    rounding_keychain: Keychain,
) -> Float[Array, "..."]:
    return lut_values_at(
        _stochastic_round_to_sorted_lut_table_indices(values, table, rounding_keychain),
        table,
    )


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _stochastic_round_to_sorted_lut_table(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
    rounding_keychain: Keychain,
) -> Float[Array, "..."]:
    return _stochastic_round_to_sorted_lut_table_impl(values, table, rounding_keychain)


def _stochastic_round_to_sorted_lut_table_fwd(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
    rounding_keychain: Keychain,
) -> tuple[Float[Array, "..."], tuple[Float[Array, "..."], Float[Array, " levels"]]]:
    rounded_values = _stochastic_round_to_sorted_lut_table_impl(values, table, rounding_keychain)
    return rounded_values, (values, table)


def _stochastic_round_to_sorted_lut_table_bwd(
    rounding_keychain: Keychain,  # noqa: ARG001
    residuals: tuple[Float[Array, "..."], Float[Array, " levels"]],
    gradients: Float[Array, "..."],
) -> tuple[Float[Array, "..."], None]:
    values, table = residuals
    return _mask_sorted_lut_table_gradients(values, gradients, table), None


_stochastic_round_to_sorted_lut_table.defvjp(
    _stochastic_round_to_sorted_lut_table_fwd,
    _stochastic_round_to_sorted_lut_table_bwd,
)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def stochastic_round_to_sorted_lut_table(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
    *,
    keychain: Keychain,
) -> Float[Array, "..."]:
    return _stochastic_round_to_sorted_lut_table(values, table, keychain)


def round_to_sorted_lut_table(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
    *,
    keychain: Keychain | None,
    gradient_estimator: GradientEstimator,
) -> Float[Array, "..."]:
    if gradient_estimator == GradientEstimator.DETERMINISTIC_ROUNDING:
        return deterministic_round_to_sorted_lut_table(values, table)
    if gradient_estimator == GradientEstimator.STOCHASTIC_ROUNDING:
        if keychain is None:
            raise ValueError("Stochastic LUT rounding requires a keychain.")
        return stochastic_round_to_sorted_lut_table(values, table, keychain=keychain)
    if gradient_estimator == GradientEstimator.LOCAL_ADDITIVE_NOISE:
        raise ValueError("Local additive noise is not implemented.")
    raise ValueError(f"Unsupported gradient estimator {gradient_estimator}")


def _minifloat_dtype_name(dtype: DTypeLike) -> str:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.dtype(jnp.float4_e2m1fn):
        return dtype.name
    if dtype == jnp.dtype(jnp.float8_e4m3fn):
        return dtype.name
    if dtype == jnp.dtype(jnp.float8_e8m0fnu):
        return dtype.name
    raise ValueError(f"Unsupported minifloat dtype {dtype}")


@cache
def _minifloat_lut_values(dtype_name: str) -> tuple[float, ...]:
    if dtype_name == jnp.dtype(jnp.float4_e2m1fn).name:
        positive_values = [0.0]
        for exponent in range(4):
            for mantissa in range(2):
                if exponent == 0:
                    if mantissa != 0:
                        positive_values.append(mantissa / 2)
                else:
                    positive_values.append(2.0 ** (exponent - 1) * (1 + mantissa / 2))
    elif dtype_name == jnp.dtype(jnp.float8_e4m3fn).name:
        positive_values = [0.0]
        for mantissa in range(1, 8):
            positive_values.append(mantissa / 512)
        for exponent in range(1, 15):
            for mantissa in range(8):
                positive_values.append(2.0 ** (exponent - 7) * (1 + mantissa / 8))
        for mantissa in range(7):
            positive_values.append(2.0 ** (15 - 7) * (1 + mantissa / 8))
    elif dtype_name == jnp.dtype(jnp.float8_e8m0fnu).name:
        return tuple(2.0**exponent for exponent in range(-_E8M0_BIAS, _E8M0_BIAS + 1))
    else:
        raise ValueError(f"Unsupported minifloat dtype {dtype_name}")

    values = set()
    for positive_value in positive_values:
        values.add(positive_value)
        values.add(-positive_value)
    return tuple(sorted(values))


def _minifloat_lut(dtype: DTypeLike, value_dtype: DTypeLike) -> Float[Array, " levels"]:
    return jnp.array(_minifloat_lut_values(_minifloat_dtype_name(dtype)), dtype=value_dtype)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def deterministic_round_to_minifloat(
    values: Float[Array, "..."],
    *,
    dtype: DTypeLike,
) -> Float[Array, "..."]:
    return deterministic_round_to_sorted_lut_table(
        values,
        _minifloat_lut(dtype, values.dtype),
    )


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def stochastic_round_to_minifloat(
    values: Float[Array, "..."],
    *,
    dtype: DTypeLike,
    keychain: Keychain,
) -> Float[Array, "..."]:
    return stochastic_round_to_sorted_lut_table(
        values,
        _minifloat_lut(dtype, values.dtype),
        keychain=keychain,
    )


def round_to_minifloat(
    values: Float[Array, "..."],
    *,
    dtype: DTypeLike,
    keychain: Keychain | None,
    gradient_estimator: GradientEstimator,
) -> Float[Array, "..."]:
    if gradient_estimator == GradientEstimator.DETERMINISTIC_ROUNDING:
        return deterministic_round_to_minifloat(values, dtype=dtype)
    if gradient_estimator == GradientEstimator.STOCHASTIC_ROUNDING:
        if keychain is None:
            raise ValueError("Stochastic minifloat rounding requires a keychain.")
        return stochastic_round_to_minifloat(values, dtype=dtype, keychain=keychain)
    if gradient_estimator == GradientEstimator.LOCAL_ADDITIVE_NOISE:
        raise ValueError("Local additive noise is not implemented.")
    raise ValueError(f"Unsupported gradient estimator {gradient_estimator}")


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def pack_e4m3_scales(
    values: Float[Array, "..."],
    *,
    out_sharding: NamedSharding | None = None,
) -> Float[Array, "..."]:
    if out_sharding is not None:
        values = with_sharding(values, out_sharding)
    return deterministic_round_to_minifloat(
        values,
        dtype=jnp.float8_e4m3fn,
    ).astype(jnp.float8_e4m3fn)


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def pack_e8m0_scales(
    values: Float[Array, "..."],
    *,
    out_sharding: NamedSharding | None = None,
) -> UInt8[Array, "..."]:
    if out_sharding is not None:
        values = with_sharding(values, out_sharding)
    values = jnp.maximum(values.astype(jnp.float32), jnp.float32(2.0**-127))
    rounded_values = deterministic_round_to_minifloat(values, dtype=jnp.float8_e8m0fnu)
    exponents = jnp.clip(jnp.round(jnp.log2(rounded_values)).astype(jnp.int32), -_E8M0_BIAS, _E8M0_BIAS)
    return (exponents + _E8M0_BIAS).astype(jnp.uint8)


def e8m0_scale_values(
    scale_exponents: UInt8[Array, "..."],
    dtype: DTypeLike,
) -> Float[Array, "..."]:
    exponents = scale_exponents.astype(jnp.int32) - _E8M0_BIAS
    return jnp.ldexp(jnp.ones_like(scale_exponents, dtype=dtype), exponents)


def _stochastic_round_to_unsigned_grid_impl(
    values: Float[Array, "..."],
    *,
    bits: int,
    keychain: Keychain,
) -> Float[Array, "..."]:
    out_sharding = sharding_of(values)
    clipped_values = jnp.clip(values, 0, unsigned_qmax(bits))
    clipped_values = with_sharding(clipped_values, out_sharding)
    lower_bins = jnp.floor(clipped_values)
    upper_probability = clipped_values - lower_bins
    samples = jax.random.uniform(
        keychain.batch_key,
        clipped_values.shape,
        dtype=clipped_values.dtype,
        out_sharding=out_sharding,
    )
    upper_samples = samples < upper_probability
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
