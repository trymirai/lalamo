from collections.abc import Callable
from dataclasses import dataclass
from math import erfc, exp, log2, pi, sqrt
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.lax import DotAlgorithmPreset

from lalamo.compressed.int import IntSpec
from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.compressed.utils import yaqa as yaqa_module
from lalamo.compressed.utils.yaqa import (
    ConvergenceError,
    _update_rounded_blocks,
    yaqa_round_blockwise,
    yaqa_round_fixpoint,
)
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

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _round_weights(weights: jax.Array) -> jax.Array:
    return jnp.round(jnp.clip(weights, -1, 1))


def _shard_weights(weights: jax.Array) -> jax.Array:
    return jax.device_put(weights, make_test_sharding_config().resolve_sharding((None,) * weights.ndim))


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
        is_sharded: bool = True,
    ) -> FullPrecisionMatrix:
        return FullPrecisionSpec().compress(
            _round_weights(weights),
            implementation=implementation,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
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
        is_sharded: bool = True,
    ) -> FullPrecisionMatrix:
        block_mean = jnp.mean(weights, axis=(-2, -1), keepdims=True)
        return FullPrecisionSpec().compress(
            jnp.broadcast_to(block_mean, weights.shape),
            implementation=implementation,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
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
        is_sharded: bool = True,
    ) -> FullPrecisionMatrix:
        value = 1 if implementation == CompressionImplementation.INFERENCE else -1
        return FullPrecisionSpec().compress(
            jnp.full_like(weights, value),
            implementation=implementation,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
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
    return _shard_weights(1.2 * jax.random.normal(jax.random.key(0), (64, 64), dtype=jnp.float32))


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
    weights = _shard_weights(jnp.stack([_weights(), _weights() * 0.5]))
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
    weights = _shard_weights(jnp.zeros((4, 4), dtype=jnp.float32))

    result = yaqa_round_fixpoint(
        weights,
        Preconditioner.identity(),
        _ImplementationSensitiveSpec(),
        sharding_config=make_test_sharding_config(),
    )

    assert_close(result=result, reference=jnp.ones_like(weights))


def test_yaqa_round_blockwise_quantizes_each_block_separately() -> None:
    weights = _shard_weights(jnp.arange(16, dtype=jnp.float32).reshape(4, 4))
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
    weights = _shard_weights(jnp.zeros((4, 4), dtype=jnp.float32))

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
    weights = _shard_weights(jnp.stack([_weights(), _weights() * 0.5]))
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


def _loop_update_rounded_blocks(
    quantized_weights: jax.Array,
    rounded_blocks: jax.Array,
    block_starts: jax.Array,
    valid_blocks: jax.Array,
) -> jax.Array:
    """The pre-scatter inner loop, kept as an independent write-order oracle."""

    def update_block(block_index: jax.Array, current: jax.Array) -> jax.Array:
        updated = jax.lax.dynamic_update_slice(
            current,
            rounded_blocks[block_index],
            (block_starts[block_index, 0], block_starts[block_index, 1]),
        )
        return jnp.where(valid_blocks[block_index], updated, current)

    return jax.lax.fori_loop(0, rounded_blocks.shape[0], update_block, quantized_weights)


def _assert_bitwise_equal(result: jax.Array, reference: jax.Array) -> None:
    assert result.shape == reference.shape
    assert result.dtype == reference.dtype
    np.testing.assert_array_equal(
        np.ascontiguousarray(result).view(np.uint8),
        np.ascontiguousarray(reference).view(np.uint8),
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("valid_count", [0, 1, 3])
def test_yaqa_block_scatter_drops_padded_boundary_windows(dtype: jnp.dtype, valid_count: int) -> None:
    weights = _shard_weights(jnp.arange(36, dtype=jnp.float32).reshape(6, 6).astype(dtype))
    # All starts are the last in-bounds window, matching clamped padding at a
    # partial anti-diagonal. Invalid writes have different values and NaNs.
    starts = _shard_weights(jnp.asarray([[4, 4], [4, 4], [4, 4]], dtype=jnp.int32))
    if valid_count == 3:
        starts = _shard_weights(jnp.asarray([[0, 4], [2, 2], [4, 0]], dtype=jnp.int32))
    valid = _shard_weights(jnp.arange(3) < valid_count)
    with jax.debug_nans(new_val=False):
        updates = _shard_weights(
            jnp.asarray(
                [[[100.0, 101.0], [102.0, 103.0]], [[200.0, 201.0], [202.0, 203.0]], [[300.0, 301.0], [302.0, 303.0]]],
                dtype=dtype,
            ),
        )
        updates = jnp.where(valid[:, None, None], updates, jnp.asarray(jnp.nan, dtype=dtype))
        result = jax.jit(_update_rounded_blocks)(weights, updates, starts, valid)
        reference = jax.jit(_loop_update_rounded_blocks)(weights, updates, starts, valid)

    _assert_bitwise_equal(result, reference)
    assert result.sharding.is_fully_replicated
    assert bool(jnp.all(jnp.isfinite(result)))


def _small_positive_definite_block(size: int, seed: int) -> jax.Array:
    factors = jax.random.normal(jax.random.key(seed), (size, size), dtype=jnp.float32)
    return factors @ factors.T / size + 0.5 * jnp.identity(size, dtype=jnp.float32)


def _blockwise_with_loop_reference(
    weights: jax.Array,
    preconditioner: Preconditioner,
    spec: QuantizedSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[jax.Array, jax.Array]:
    sharding = make_test_sharding_config()
    result = jax.jit(
        lambda w, p: yaqa_round_blockwise(w, p, spec, sharding_config=sharding),
    )(weights, preconditioner)
    oracle_calls = 0

    def traced_reference(
        operand: jax.Array,
        updates: jax.Array,
        starts: jax.Array,
        valid: jax.Array,
    ) -> jax.Array:
        nonlocal oracle_calls
        oracle_calls += 1
        return _loop_update_rounded_blocks(operand, updates, starts, valid)

    with monkeypatch.context() as context:
        context.setattr(yaqa_module, "_update_rounded_blocks", traced_reference)
        # A different jit closure forces tracing through the preserved updater,
        # rather than reusing the optimized executable from the first call.
        reference = jax.jit(
            lambda w, p: yaqa_round_blockwise(w, p, spec, sharding_config=sharding),
        )(weights, preconditioner)
    assert oracle_calls > 0
    return result, reference


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize(
    ("shape", "spec", "feedback"),
    [
        ((8, 20), _RoundSpec(), "input"),
        ((20, 8), _RoundSpec(), "output"),
        ((6, 10), _BlockMeanSpec(), "two_sided"),
        ((3, 128), IntSpec(bits=4, group_size=64), "two_sided"),
    ],
)
def test_yaqa_block_scatter_matches_loop_for_rectangular_preconditioned_rounding(
    dtype: jnp.dtype,
    shape: tuple[int, int],
    spec: QuantizedSpec,
    feedback: Literal["input", "output", "two_sided"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dim, input_dim = shape
    weights = _shard_weights(jax.random.normal(jax.random.key(41), shape, dtype=jnp.float32).astype(dtype))
    preconditioner = Preconditioner.init(
        input_block=_small_positive_definite_block(input_dim, 43) if feedback != "output" else None,
        output_block=_small_positive_definite_block(output_dim, 47) if feedback != "input" else None,
    )
    result, reference = _blockwise_with_loop_reference(weights, preconditioner, spec, monkeypatch)

    _assert_bitwise_equal(result, reference)
    assert result.sharding == weights.sharding


def test_yaqa_block_scatter_matches_loop_under_batched_vmap(monkeypatch: pytest.MonkeyPatch) -> None:
    weights = _shard_weights(jax.random.normal(jax.random.key(53), (2, 6, 10), dtype=jnp.float32).astype(jnp.bfloat16))
    preconditioner = Preconditioner.init(
        input_block=jnp.stack([_small_positive_definite_block(10, 59), _small_positive_definite_block(10, 61)]),
        output_block=jnp.stack([_small_positive_definite_block(6, 67), _small_positive_definite_block(6, 71)]),
    )
    result, reference = _blockwise_with_loop_reference(weights, preconditioner, _BlockMeanSpec(), monkeypatch)

    _assert_bitwise_equal(result, reference)
    assert result.sharding == weights.sharding
