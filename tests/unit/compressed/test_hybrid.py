from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed.hybrid import HybridMatrix, HybridSpec
from lalamo.compressed.low_rank import LowRankSpec
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    WeightMatrixSpec,
)
from tests.common import assert_close


def _weights(output_dim: int = 32, input_dim: int = 64) -> jax.Array:
    shape = (output_dim, input_dim)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 17) / 19


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _ragged_dot_reference(matrix: HybridMatrix, vectors: jax.Array, group_sizes: jax.Array) -> jax.Array:
    weights = jnp.asarray(jax.device_get(matrix.decompress()))
    vectors = jnp.asarray(jax.device_get(vectors))
    group_sizes = jnp.asarray(jax.device_get(group_sizes))
    expert_indices = jnp.repeat(
        jnp.arange(weights.shape[0]),
        group_sizes,
        total_repeat_length=vectors.shape[0],
    )
    return jnp.einsum("toi,ti->to", weights[expert_indices], vectors)


@dataclass(frozen=True)
class _PreconditionerRecordingSpec(WeightMatrixSpec):
    calls: list[Preconditioner | None]

    def compress(
        self,
        weights: jax.Array,
        *,
        key: jax.Array | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> FullPrecisionMatrix:
        self.calls.append(preconditioner)
        return FullPrecisionSpec().compress(weights, implementation=implementation)


def test_hybrid_spec_roundtrips_json() -> None:
    spec = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=LowRankSpec(rank=2),
        incoherence_block_size=32,
    )

    restored = WeightMatrixSpec.from_json(spec.to_json())

    assert restored == spec


def test_hybrid_compress_requires_key_when_incoherence_is_enabled() -> None:
    spec = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
    )

    with pytest.raises(ValueError, match="random incoherence signs"):
        spec.compress(_weights())


def test_hybrid_compress_without_incoherence_does_not_require_key() -> None:
    weights = _weights()
    spec = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=None,
    )

    matrix = spec.compress(weights)

    assert isinstance(matrix, HybridMatrix)
    assert matrix.incoherence_signs is None
    _assert_close(result=matrix.decompress(), reference=weights)


def test_hybrid_incoherence_decompress_restores_original_basis() -> None:
    weights = _weights()
    spec = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
    )

    matrix = spec.compress(weights, key=jax.random.key(0))

    assert isinstance(matrix, HybridMatrix)
    assert matrix.incoherence_signs is not None
    assert matrix.incoherence_signs.input_signs.shape == (64,)
    assert matrix.incoherence_signs.output_signs.shape == (32,)
    _assert_close(result=matrix.decompress(), reference=weights)


def test_hybrid_incoherence_dot_matches_original_weights() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
    ).compress(weights, key=jax.random.key(1))
    vector = (jnp.arange(64, dtype=jnp.float32) - 3) / 7

    result = matrix.dot(vector, keychain=Keychain.init(0))

    _assert_close(result=result, reference=weights @ vector)


def test_hybrid_incoherence_transposed_dot_matches_original_weights() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
    ).compress(weights, key=jax.random.key(2))
    vector = (jnp.arange(32, dtype=jnp.float32) + 5) / 11

    result = matrix.dot(vector, keychain=Keychain.init(1), transposed=True)

    _assert_close(result=result, reference=weights.T @ vector)


def test_hybrid_incoherence_ragged_dot_matches_original_weights() -> None:
    weights = jnp.stack([_weights(), _weights() / 3])
    matrix = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
    ).compress(weights, key=jax.random.key(3))
    vectors = (jnp.arange(3 * 64, dtype=jnp.float32).reshape(3, 64) - 3) / 7
    group_sizes = jnp.array([1, 2], dtype=jnp.int32)

    result = matrix.ragged_dot(vectors, group_sizes, keychain=Keychain.init(2))

    _assert_close(result=result, reference=_ragged_dot_reference(matrix, vectors, group_sizes))


def test_hybrid_adapter_compresses_residual_in_incoherent_basis() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=LowRankSpec(rank=1),
        adapter_spec=FullPrecisionSpec(),
        incoherence_block_size=32,
    ).compress(weights, key=jax.random.key(3))

    _assert_close(result=matrix.decompress(), reference=weights)


def test_hybrid_compress_reuses_preconditioner_for_quantization_and_adapter() -> None:
    quantization_calls: list[Preconditioner | None] = []
    adapter_calls: list[Preconditioner | None] = []
    preconditioner = Preconditioner.init(input_block=jnp.eye(64, dtype=jnp.float32))
    spec = HybridSpec(
        quantization_spec=_PreconditionerRecordingSpec(quantization_calls),
        adapter_spec=_PreconditionerRecordingSpec(adapter_calls),
        incoherence_block_size=None,
    )

    spec.compress(_weights(), preconditioner=preconditioner)

    assert len(quantization_calls) == 1
    assert len(adapter_calls) == 1
    assert quantization_calls[0] is preconditioner
    assert adapter_calls[0] is preconditioner
