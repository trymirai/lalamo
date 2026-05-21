from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax import ShapeDtypeStruct
from jaxtyping import Array, Float

from lalamo.compressed.utils.rounding import (
    deterministic_round_to_minifloat,
    deterministic_round_to_sorted_lut_table,
    deterministic_round_to_unsigned_grid,
    pack_e4m3_scales,
    pack_e8m0_scales,
    round_to_minifloat,
    round_to_sorted_lut_table,
    round_to_unsigned_grid,
    stochastic_round_to_minifloat,
    stochastic_round_to_sorted_lut_table,
    stochastic_round_to_unsigned_grid,
)
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import GradientEstimator
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config


def _expected_straight_through_gradients(
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
    *,
    bits: int,
) -> Float[Array, "..."]:
    qmax = (2**bits) - 1
    inside_range = (values >= 0) & (values <= qmax)
    below_range_with_allowed_gradient = (values < 0) & (gradients > 0)
    above_range_with_allowed_gradient = (values > qmax) & (gradients < 0)
    return jnp.where(
        inside_range | below_range_with_allowed_gradient | above_range_with_allowed_gradient,
        gradients,
        0,
    )


def _expected_lut_straight_through_gradients(
    values: Float[Array, "..."],
    gradients: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> Float[Array, "..."]:
    inside_range = (values >= table[0]) & (values <= table[-1])
    below_range_with_allowed_gradient = (values < table[0]) & (gradients > 0)
    above_range_with_allowed_gradient = (values > table[-1]) & (gradients < 0)
    return jnp.where(
        inside_range | below_range_with_allowed_gradient | above_range_with_allowed_gradient,
        gradients,
        0,
    )


def _shard(values: Array) -> Array:
    return jax.device_put(values, make_test_sharding_config().resolve_sharding((None,) * values.ndim))


@pytest.mark.parametrize(
    ("bits", "values", "expected"),
    [
        (
            2,
            [-1.0, 0.0, 0.49, 0.51, 1.49, 1.51, 2.49, 2.51, 3.0, 4.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        ),
        (
            4,
            [-1.0, 0.0, 7.49, 7.51, 14.49, 14.51, 15.0, 16.0],
            [0.0, 0.0, 7.0, 8.0, 14.0, 15.0, 15.0, 15.0],
        ),
    ],
)
def test_deterministic_round_to_unsigned_grid_clips_and_rounds(
    bits: int,
    values: list[float],
    expected: list[float],
) -> None:
    input_values = jnp.array(values, dtype=jnp.float32)
    expected_values = jnp.array(expected, dtype=jnp.float32)

    result = deterministic_round_to_unsigned_grid(input_values, bits=bits)

    assert_close(result=result, reference=expected_values)


def test_stochastic_round_to_unsigned_grid_clips_exact_bins() -> None:
    values = _shard(jnp.array([-1.0, 0.0, 1.0, 7.0, 15.0, 16.0], dtype=jnp.float32))
    expected = jnp.array([0.0, 0.0, 1.0, 7.0, 15.0, 15.0], dtype=jnp.float32)

    result = stochastic_round_to_unsigned_grid(
        values, bits=4, keychain=Keychain.init(11, sharding_config=make_test_sharding_config())
    )

    assert_close(result=result, reference=expected)


def test_stochastic_round_to_unsigned_grid_samples_adjacent_bins() -> None:
    values = _shard(jnp.array([0.125, 1.25, 7.5, 14.875], dtype=jnp.float32))
    lower_bins = jnp.floor(values)
    upper_bins = lower_bins + 1

    result = stochastic_round_to_unsigned_grid(
        values, bits=4, keychain=Keychain.init(12, sharding_config=make_test_sharding_config())
    )

    assert bool(jnp.all((result == lower_bins) | (result == upper_bins)))
    assert bool(jnp.all(result >= 0))
    assert bool(jnp.all(result <= 15))


def test_stochastic_round_to_unsigned_grid_is_unbiased_for_fractional_bins() -> None:
    num_samples = 32_768
    expected_means = jnp.array([1.125, 3.5, 14.875], dtype=jnp.float32)
    values = _shard(jnp.broadcast_to(expected_means[:, None], (expected_means.size, num_samples)))

    result = stochastic_round_to_unsigned_grid(
        values, bits=4, keychain=Keychain.init(13, sharding_config=make_test_sharding_config())
    )

    assert_close(result=jnp.mean(result, axis=-1), reference=expected_means, atol=0.015, rtol=0.0)


def test_round_to_unsigned_grid_selects_deterministic_rounding() -> None:
    values = jnp.array([-1.0, 0.51, 14.51, 16.0], dtype=jnp.float32)

    result = round_to_unsigned_grid(
        values,
        bits=4,
        keychain=Keychain.init(14, sharding_config=make_test_sharding_config()),
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    expected = deterministic_round_to_unsigned_grid(values, bits=4)

    assert_close(result=result, reference=expected)


def test_round_to_unsigned_grid_selects_stochastic_rounding() -> None:
    values = _shard(jnp.array([0.125, 1.25, 7.5, 14.875], dtype=jnp.float32))
    keychain = Keychain.init(15, sharding_config=make_test_sharding_config())

    result = round_to_unsigned_grid(
        values,
        bits=4,
        keychain=keychain,
        gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING,
    )
    expected = stochastic_round_to_unsigned_grid(values, bits=4, keychain=keychain)

    assert_close(result=result, reference=expected)


def test_round_to_unsigned_grid_rejects_local_additive_noise() -> None:
    values = jnp.array([1.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="Local additive noise is not implemented"):
        round_to_unsigned_grid(
            values,
            bits=4,
            keychain=Keychain.init(16, sharding_config=make_test_sharding_config()),
            gradient_estimator=GradientEstimator.LOCAL_ADDITIVE_NOISE,
        )


def test_deterministic_round_to_sorted_lut_table_uses_nearest_values() -> None:
    values = _shard(jnp.array([-2.0, -0.4, 0.75, 1.6, 2.0, 4.0], dtype=jnp.float32))
    table = jnp.array([-1.0, 0.0, 1.0, 3.0], dtype=jnp.float32)
    expected = jnp.array([-1.0, 0.0, 1.0, 1.0, 1.0, 3.0], dtype=jnp.float32)

    result = deterministic_round_to_sorted_lut_table(values, table)

    assert_close(result=result, reference=expected)


def test_stochastic_round_to_sorted_lut_table_is_distance_weighted() -> None:
    values = _shard(jnp.broadcast_to(jnp.array([[0.5], [2.25]], dtype=jnp.float32), (2, 32_768)))
    table = jnp.array([0.0, 2.0, 3.0], dtype=jnp.float32)

    result = stochastic_round_to_sorted_lut_table(
        values, table, keychain=Keychain.init(21, sharding_config=make_test_sharding_config())
    )

    assert_close(result=jnp.mean(result, axis=-1), reference=jnp.array([0.5, 2.25]), atol=0.02, rtol=0.0)


def test_round_to_sorted_lut_table_selects_stochastic_rounding() -> None:
    values = _shard(jnp.array([0.25, 0.75], dtype=jnp.float32))
    table = jnp.array([0.0, 1.0], dtype=jnp.float32)
    keychain = Keychain.init(22, sharding_config=make_test_sharding_config())

    result = round_to_sorted_lut_table(
        values,
        table,
        keychain=keychain,
        gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING,
    )
    expected = stochastic_round_to_sorted_lut_table(values, table, keychain=keychain)

    assert_close(result=result, reference=expected)


def test_minifloat_rounding_uses_nearest_representable_values() -> None:
    values = _shard(jnp.array([0.01, 1.03, 1.09, 447.0], dtype=jnp.float32))
    expected = values.astype(jnp.float8_e4m3fn).astype(jnp.float32)

    result = deterministic_round_to_minifloat(values, dtype=jnp.float8_e4m3fn)

    assert_close(result=result, reference=expected)


def test_stochastic_round_to_minifloat_samples_adjacent_values() -> None:
    values = _shard(jnp.broadcast_to(jnp.array(2.5, dtype=jnp.float32), (32_768,)))

    result = stochastic_round_to_minifloat(
        values, dtype=jnp.float4_e2m1fn, keychain=Keychain.init(23, sharding_config=make_test_sharding_config())
    )

    assert bool(jnp.all((result == 2.0) | (result == 3.0)))
    assert_close(result=jnp.mean(result), reference=jnp.array(2.5), atol=0.02, rtol=0.0)


def test_round_to_minifloat_selects_deterministic_rounding() -> None:
    values = _shard(jnp.array([1.03, 1.09], dtype=jnp.float32))

    result = round_to_minifloat(
        values,
        dtype=jnp.float8_e4m3fn,
        keychain=Keychain.init(24, sharding_config=make_test_sharding_config()),
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    expected = deterministic_round_to_minifloat(values, dtype=jnp.float8_e4m3fn)

    assert_close(result=result, reference=expected)


def test_e4m3_and_e8m0_scale_packing_uses_nearest_minifloat() -> None:
    e4m3_values = _shard(jnp.array([0.01, 1.03, 1.09], dtype=jnp.float32))
    e8m0_values = _shard(jnp.array([1.49, 1.5, 2.9, 3.0], dtype=jnp.float32))

    e4m3_result = pack_e4m3_scales(e4m3_values)
    e8m0_result = pack_e8m0_scales(e8m0_values)

    assert e4m3_result.dtype == jnp.float8_e4m3fn
    assert_close(
        result=e4m3_result.astype(jnp.float32),
        reference=e4m3_values.astype(jnp.float8_e4m3fn).astype(jnp.float32),
    )
    assert jnp.array_equal(e8m0_result, jnp.array([127, 127, 128, 128], dtype=jnp.uint8))


def test_rounding_shape_dtype_struct_inputs_return_dummy_arrays() -> None:
    values = dummy_array((2, 3), jnp.float16, make_sharding((None, None)))
    table = jnp.array([0, 1, 2], dtype=jnp.float16)

    deterministic_result = deterministic_round_to_unsigned_grid(values, bits=4)
    stochastic_result = stochastic_round_to_unsigned_grid(
        values,
        bits=4,
        keychain=Keychain.init(17, sharding_config=make_test_sharding_config()),
    )
    dispatched_result = round_to_unsigned_grid(
        values,
        bits=4,
        keychain=Keychain.init(18, sharding_config=make_test_sharding_config()),
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    lut_result = round_to_sorted_lut_table(
        values,
        table,
        keychain=Keychain.init(19, sharding_config=make_test_sharding_config()),
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )
    minifloat_result = round_to_minifloat(
        values,
        dtype=jnp.float8_e4m3fn,
        keychain=Keychain.init(20, sharding_config=make_test_sharding_config()),
        gradient_estimator=GradientEstimator.DETERMINISTIC_ROUNDING,
    )

    for result in (deterministic_result, stochastic_result, dispatched_result, lut_result, minifloat_result):
        assert isinstance(result, ShapeDtypeStruct)
        assert result.shape == values.shape
        assert result.dtype == values.dtype


def test_deterministic_round_to_unsigned_grid_backward_masks_straight_through_gradients() -> None:
    values = jnp.array([-1.0, -1.0, 0.0, 7.25, 15.0, 16.0, 16.0], dtype=jnp.float32)
    incoming_gradients = jnp.array([2.0, -2.0, -3.0, 4.0, -5.0, 2.0, -2.0], dtype=jnp.float32)
    expected = _expected_straight_through_gradients(values, incoming_gradients, bits=4)

    def loss(input_values: Float[Array, "..."]) -> Float[Array, ""]:
        return jnp.sum(deterministic_round_to_unsigned_grid(input_values, bits=4) * incoming_gradients)

    with jax.set_mesh(make_test_sharding_config().mesh):
        result = jax.grad(loss)(values)

    assert_close(result=result, reference=expected)


def test_stochastic_round_to_unsigned_grid_backward_masks_straight_through_gradients() -> None:
    values = _shard(jnp.array([-1.0, -1.0, 0.0, 7.25, 15.0, 16.0, 16.0], dtype=jnp.float32))
    incoming_gradients = jnp.array([2.0, -2.0, -3.0, 4.0, -5.0, 2.0, -2.0], dtype=jnp.float32)
    expected = _expected_straight_through_gradients(values, incoming_gradients, bits=4)
    keychain = Keychain.init(19, sharding_config=make_test_sharding_config())

    def loss(input_values: Float[Array, "..."]) -> Float[Array, ""]:
        rounded_values = stochastic_round_to_unsigned_grid(input_values, bits=4, keychain=keychain)
        return jnp.sum(rounded_values * incoming_gradients)

    with jax.set_mesh(make_test_sharding_config().mesh):
        result = jax.grad(loss)(values)

    assert_close(result=result, reference=expected)


def test_round_to_unsigned_grid_backward_masks_straight_through_gradients_under_jit() -> None:
    values = _shard(jnp.array([-1.0, 0.0, 7.25, 15.0, 16.0], dtype=jnp.float32))
    incoming_gradients = jnp.array([2.0, -3.0, 4.0, -5.0, -2.0], dtype=jnp.float32)
    expected = _expected_straight_through_gradients(values, incoming_gradients, bits=4)
    keychain = Keychain.init(20, sharding_config=make_test_sharding_config())

    def loss(input_values: Float[Array, "..."]) -> Float[Array, ""]:
        rounded_values = round_to_unsigned_grid(
            input_values,
            bits=4,
            keychain=keychain,
            gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING,
        )
        return jnp.sum(rounded_values * incoming_gradients)

    with jax.set_mesh(make_test_sharding_config().mesh):
        result = jax.jit(jax.grad(loss))(values)

    assert_close(result=result, reference=expected)


@pytest.mark.parametrize(
    "round_fn",
    [
        deterministic_round_to_sorted_lut_table,
        lambda values, table: stochastic_round_to_sorted_lut_table(
            values, table, keychain=Keychain.init(25, sharding_config=make_test_sharding_config())
        ),
        lambda values, table: round_to_sorted_lut_table(
            values,
            table,
            keychain=Keychain.init(26, sharding_config=make_test_sharding_config()),
            gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING,
        ),
    ],
)
def test_round_to_sorted_lut_table_backward_masks_straight_through_gradients(
    round_fn: Callable[[Float[Array, "..."], Float[Array, " levels"]], Float[Array, "..."]],
) -> None:
    values = _shard(jnp.array([-2.0, -2.0, -1.0, 0.5, 3.0, 4.0, 4.0], dtype=jnp.float32))
    table = jnp.array([-1.0, 0.0, 3.0], dtype=jnp.float32)
    incoming_gradients = jnp.array([2.0, -2.0, -3.0, 4.0, -5.0, 2.0, -2.0], dtype=jnp.float32)
    expected = _expected_lut_straight_through_gradients(values, incoming_gradients, table)

    def loss(input_values: Float[Array, "..."]) -> Float[Array, ""]:
        return jnp.sum(round_fn(input_values, table) * incoming_gradients)

    with jax.set_mesh(make_test_sharding_config().mesh):
        result = jax.grad(loss)(values)

    assert_close(result=result, reference=expected)
