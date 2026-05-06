import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset

from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.weight_matrix import Preconditioner
from tests.common import assert_close


def _round_weights(weights: jax.Array) -> jax.Array:
    return jnp.round(jnp.clip(weights, -1, 1))


def _weights() -> jax.Array:
    return jnp.array(
        [
            [-0.184103, 0.101259, -0.162245, -0.184811],
            [1.192088, 0.176838, 1.451657, 1.025973],
            [-0.342661, 0.419872, -0.781515, -0.290960],
        ],
        dtype=jnp.float32,
    )


def _input_block() -> jax.Array:
    return jnp.array(
        [
            [1.0, 0.7, -0.2, 0.1],
            [0.7, 1.4, 0.3, -0.1],
            [-0.2, 0.3, 1.1, 0.5],
            [0.1, -0.1, 0.5, 1.3],
        ],
        dtype=jnp.float32,
    )


def _output_block() -> jax.Array:
    return jnp.array(
        [
            [1.2, -0.5, 0.2],
            [-0.5, 1.5, 0.4],
            [0.2, 0.4, 1.1],
        ],
        dtype=jnp.float32,
    )


def _output_only_weights() -> jax.Array:
    return jnp.array(
        [
            [1.343001, 1.435740, -0.503126, -0.093995],
            [0.209666, -1.003491, -0.569416, 0.568442],
            [0.740300, -0.986956, 1.456061, -1.424152],
        ],
        dtype=jnp.float32,
    )


def _quadratic_objective(
    weights: jax.Array,
    rounded_weights: jax.Array,
    preconditioner: Preconditioner,
) -> jax.Array:
    error = weights - rounded_weights
    input_block = preconditioner.input_block
    if input_block is None:
        input_block = jnp.identity(weights.shape[1], dtype=weights.dtype)

    output_block = preconditioner.output_block
    if output_block is None:
        output_block = jnp.identity(weights.shape[0], dtype=weights.dtype)

    with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
        return jnp.trace(error @ input_block @ error.T @ output_block)


def _assert_yaqa_reduces_quadratic_objective(weights: jax.Array, preconditioner: Preconditioner) -> None:
    baseline_weights = _round_weights(weights)
    yaqa_weights = yaqa_round_weights(
        weights,
        preconditioner,
        _round_weights,
        input_block_size=2,
        output_block_size=1,
    )

    baseline_objective = _quadratic_objective(weights, baseline_weights, preconditioner)
    yaqa_objective = _quadratic_objective(weights, yaqa_weights, preconditioner)

    assert yaqa_objective < baseline_objective


def test_yaqa_round_weights_reduces_input_preconditioned_quadratic_objective() -> None:
    _assert_yaqa_reduces_quadratic_objective(
        _weights(),
        Preconditioner.init(input_block=_input_block()),
    )


def test_yaqa_round_weights_reduces_output_preconditioned_quadratic_objective() -> None:
    _assert_yaqa_reduces_quadratic_objective(
        _output_only_weights(),
        Preconditioner.init(output_block=_output_block()),
    )


def test_yaqa_round_weights_reduces_two_sided_preconditioned_quadratic_objective() -> None:
    _assert_yaqa_reduces_quadratic_objective(
        _weights(),
        Preconditioner.init(input_block=_input_block(), output_block=_output_block()),
    )


def test_yaqa_round_weights_with_identity_preconditioner_matches_baseline_rounding() -> None:
    weights = _weights()

    result = yaqa_round_weights(
        weights,
        Preconditioner.identity(),
        _round_weights,
        input_block_size=2,
        output_block_size=1,
    )

    assert_close(result=result, reference=_round_weights(weights))


def test_yaqa_round_weights_single_iteration_returns_rounded_weights_with_expected_shape_and_dtype() -> None:
    weights = _weights()

    result = yaqa_round_weights(
        weights,
        Preconditioner.init(input_block=_input_block(), output_block=_output_block()),
        _round_weights,
        input_block_size=2,
        output_block_size=1,
        max_iters=1,
    )

    assert result.shape == weights.shape
    assert result.dtype == weights.dtype
    assert bool(jnp.all(result == _round_weights(result)))
