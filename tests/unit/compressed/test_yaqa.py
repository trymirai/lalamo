from dataclasses import dataclass
from math import erfc, exp, log2, pi, sqrt

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset

from lalamo.compressed.utils.yaqa import yaqa_round_weights
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Preconditioner,
    QuantizedSpec,
)
from tests.common import assert_close


def _round_weights(weights: jax.Array) -> jax.Array:
    return jnp.round(jnp.clip(weights, -1, 1))


@dataclass(frozen=True)
class _RoundSpec(QuantizedSpec):
    def compress(
        self,
        weights: jax.Array,
        *,
        key: jax.Array | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> FullPrecisionMatrix:
        return FullPrecisionSpec().compress(_round_weights(weights), implementation=implementation)

    @property
    def input_block_size(self) -> int:
        return 4

    @property
    def output_block_size(self) -> int:
        return 4

    @property
    def rate(self) -> float:
        return log2(3)

    @property
    def distortion(self) -> float:
        threshold = 0.5
        density = exp(-(threshold**2) / 2) / sqrt(2 * pi)
        tail_probability = erfc(threshold / sqrt(2)) / 2
        return 1 - 4 * density + 2 * tail_probability


def _weights() -> jax.Array:
    return 1.2 * jax.random.normal(jax.random.key(0), (64, 64), dtype=jnp.float32)


def _positive_definite_block(seed: int) -> jax.Array:
    factors = jax.random.normal(jax.random.key(seed), (64, 64), dtype=jnp.float32)
    return factors @ factors.T / 64 + 0.5 * jnp.identity(64, dtype=jnp.float32)


def _input_block() -> jax.Array:
    return _positive_definite_block(11)


def _output_block() -> jax.Array:
    return _positive_definite_block(17)


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
        _RoundSpec(),
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
        _weights(),
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
        _RoundSpec(),
    )

    assert_close(result=result, reference=_round_weights(weights))


def test_yaqa_round_weights_single_iteration_returns_rounded_weights_with_expected_shape_and_dtype() -> None:
    weights = _weights()

    result = yaqa_round_weights(
        weights,
        Preconditioner.init(input_block=_input_block(), output_block=_output_block()),
        _RoundSpec(),
        max_iters=1,
    )

    assert result.shape == weights.shape
    assert result.dtype == weights.dtype
    assert bool(jnp.all(result == _round_weights(result)))


def test_yaqa_round_weights_runs_under_vmap_with_batched_preconditioners() -> None:
    weights = jnp.stack([_weights(), _weights() * 0.5])
    input_blocks = jnp.stack([_input_block(), _positive_definite_block(23)])
    output_blocks = jnp.stack([_output_block(), _positive_definite_block(29)])
    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)
    spec = _RoundSpec()

    result = yaqa_round_weights(weights, preconditioner, spec)
    reference = jnp.stack(
        [
            yaqa_round_weights(
                weights[0],
                Preconditioner.init(input_block=input_blocks[0], output_block=output_blocks[0]),
                spec,
            ),
            yaqa_round_weights(
                weights[1],
                Preconditioner.init(input_block=input_blocks[1], output_block=output_blocks[1]),
                spec,
            ),
        ],
    )

    assert_close(result=result, reference=reference)
