from collections.abc import Callable
from dataclasses import dataclass
from math import erfc, exp, log2, pi, sqrt

import jax
import jax.numpy as jnp
import pytest
from jax.lax import DotAlgorithmPreset

from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.compressed.utils.yaqa import ConvergenceError, yaqa_round_blockwise, yaqa_round_fixpoint
from lalamo.preconditioner import Preconditioner
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
)
from tests.common import assert_close
from tests.helpers import make_test_sharding_config


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
        sharding_config: ShardingConfig,
    ) -> FullPrecisionMatrix:
        return FullPrecisionSpec().compress(
            _round_weights(weights),
            implementation=implementation,
            sharding_config=sharding_config,
        )

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


@dataclass(frozen=True)
class _BlockMeanSpec(QuantizedSpec):
    def compress(
        self,
        weights: jax.Array,
        *,
        key: jax.Array | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
    ) -> FullPrecisionMatrix:
        block_mean = jnp.mean(weights, axis=(-2, -1), keepdims=True)
        return FullPrecisionSpec().compress(
            jnp.broadcast_to(block_mean, weights.shape),
            implementation=implementation,
            sharding_config=sharding_config,
        )

    @property
    def input_block_size(self) -> int:
        return 2

    @property
    def output_block_size(self) -> int:
        return 2

    @property
    def rate(self) -> float:
        return 1

    @property
    def distortion(self) -> float:
        return 1


@dataclass(frozen=True)
class _ImplementationSensitiveSpec(QuantizedSpec):
    def compress(
        self,
        weights: jax.Array,
        *,
        key: jax.Array | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
    ) -> FullPrecisionMatrix:
        value = 1 if implementation == CompressionImplementation.INFERENCE else -1
        return FullPrecisionSpec().compress(
            jnp.full_like(weights, value),
            implementation=implementation,
            sharding_config=sharding_config,
        )

    @property
    def input_block_size(self) -> int:
        return 2

    @property
    def output_block_size(self) -> int:
        return 2

    @property
    def rate(self) -> float:
        return 1

    @property
    def distortion(self) -> float:
        return 1


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
    yaqa_weights = yaqa_round_fixpoint(
        weights,
        preconditioner,
        _RoundSpec(),
        sharding_config=make_test_sharding_config(),
    )

    baseline_objective = _quadratic_objective(weights, baseline_weights, preconditioner)
    yaqa_objective = _quadratic_objective(weights, yaqa_weights, preconditioner)

    assert yaqa_objective < baseline_objective


def test_yaqa_round_fixpoint_reduces_input_preconditioned_quadratic_objective() -> None:
    _assert_yaqa_reduces_quadratic_objective(
        _weights(),
        Preconditioner.init(input_block=_input_block()),
    )


def test_yaqa_round_fixpoint_reduces_output_preconditioned_quadratic_objective() -> None:
    _assert_yaqa_reduces_quadratic_objective(
        _weights(),
        Preconditioner.init(output_block=_output_block()),
    )


def test_yaqa_round_fixpoint_reduces_two_sided_preconditioned_quadratic_objective() -> None:
    _assert_yaqa_reduces_quadratic_objective(
        _weights(),
        Preconditioner.init(input_block=_input_block(), output_block=_output_block()),
    )


def test_yaqa_round_fixpoint_with_identity_preconditioner_matches_baseline_rounding() -> None:
    weights = _weights()

    result = yaqa_round_fixpoint(
        weights,
        Preconditioner.identity(),
        _RoundSpec(),
        sharding_config=make_test_sharding_config(),
    )

    assert_close(result=result, reference=_round_weights(weights))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("round_fn", [yaqa_round_fixpoint, yaqa_round_blockwise])
def test_yaqa_rounding_preserves_input_dtype(
    dtype: jnp.dtype,
    round_fn: Callable[..., jax.Array],
) -> None:
    weights = _weights().astype(dtype)

    result = round_fn(
        weights,
        Preconditioner.identity(),
        _RoundSpec(),
        sharding_config=make_test_sharding_config(),
    )

    assert result.dtype == weights.dtype


def test_yaqa_round_fixpoint_raises_when_iteration_limit_prevents_convergence() -> None:
    weights = _weights()

    with pytest.raises(ConvergenceError, match="failed to converge"):
        yaqa_round_fixpoint(
            weights,
            Preconditioner.init(input_block=_input_block(), output_block=_output_block()),
            _RoundSpec(),
            sharding_config=make_test_sharding_config(),
            max_iters=1,
        )


def test_yaqa_round_fixpoint_runs_under_vmap_with_batched_preconditioners() -> None:
    weights = jnp.stack([_weights(), _weights() * 0.5])
    input_blocks = jnp.stack([_input_block(), _positive_definite_block(23)])
    output_blocks = jnp.stack([_output_block(), _positive_definite_block(29)])
    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)
    spec = _RoundSpec()

    result = yaqa_round_fixpoint(weights, preconditioner, spec, sharding_config=make_test_sharding_config())
    reference = jnp.stack(
        [
            yaqa_round_fixpoint(
                weights[0],
                Preconditioner.init(input_block=input_blocks[0], output_block=output_blocks[0]),
                spec,
                sharding_config=make_test_sharding_config(),
            ),
            yaqa_round_fixpoint(
                weights[1],
                Preconditioner.init(input_block=input_blocks[1], output_block=output_blocks[1]),
                spec,
                sharding_config=make_test_sharding_config(),
            ),
        ],
    )

    assert_close(result=result, reference=reference)


def test_yaqa_round_fixpoint_uses_inference_compression() -> None:
    weights = jnp.zeros((4, 4), dtype=jnp.float32)

    result = yaqa_round_fixpoint(
        weights,
        Preconditioner.identity(),
        _ImplementationSensitiveSpec(),
        sharding_config=make_test_sharding_config(),
    )

    assert_close(result=result, reference=jnp.ones_like(weights))


def test_yaqa_round_blockwise_quantizes_each_block_separately() -> None:
    weights = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    block_means = jnp.mean(weights.reshape(2, 2, 2, 2), axis=(1, 3), keepdims=True)
    expected = jnp.broadcast_to(block_means, (2, 2, 2, 2)).reshape(4, 4)

    result = yaqa_round_blockwise(
        weights,
        Preconditioner.identity(),
        _BlockMeanSpec(),
        sharding_config=make_test_sharding_config(),
    )

    assert_close(result=result, reference=expected)


def test_yaqa_round_blockwise_uses_inference_compression() -> None:
    weights = jnp.zeros((4, 4), dtype=jnp.float32)

    result = yaqa_round_blockwise(
        weights,
        Preconditioner.identity(),
        _ImplementationSensitiveSpec(),
        sharding_config=make_test_sharding_config(),
    )

    assert_close(result=result, reference=jnp.ones_like(weights))


def test_yaqa_round_blockwise_matches_fixpoint_for_two_sided_preconditioner() -> None:
    weights = _weights()
    preconditioner = Preconditioner.init(input_block=_input_block(), output_block=_output_block())
    spec = _RoundSpec()

    result = yaqa_round_blockwise(weights, preconditioner, spec, sharding_config=make_test_sharding_config())
    reference = yaqa_round_fixpoint(weights, preconditioner, spec, sharding_config=make_test_sharding_config())

    assert_close(result=result, reference=reference)


def test_yaqa_round_blockwise_runs_under_vmap_with_batched_preconditioners() -> None:
    weights = jnp.stack([_weights(), _weights() * 0.5])
    input_blocks = jnp.stack([_input_block(), _positive_definite_block(23)])
    output_blocks = jnp.stack([_output_block(), _positive_definite_block(29)])
    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)
    spec = _RoundSpec()

    result = yaqa_round_blockwise(weights, preconditioner, spec, sharding_config=make_test_sharding_config())
    reference = jnp.stack(
        [
            yaqa_round_blockwise(
                weights[0],
                Preconditioner.init(input_block=input_blocks[0], output_block=output_blocks[0]),
                spec,
                sharding_config=make_test_sharding_config(),
            ),
            yaqa_round_blockwise(
                weights[1],
                Preconditioner.init(input_block=input_blocks[1], output_block=output_blocks[1]),
                spec,
                sharding_config=make_test_sharding_config(),
            ),
        ],
    )

    assert_close(result=result, reference=reference)
