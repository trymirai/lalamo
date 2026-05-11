from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Int, Key

from lalamo.compressed.hybrid import HybridMatrix, HybridSpec, IncoherenceProcessingMode
from lalamo.compressed.low_rank import LowRankSpec
from lalamo.compressed.utils.hadamard import hadamard_transform
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    WeightMatrixSpec,
)
from tests.common import assert_close


def _weights(output_dim: int = 32, input_dim: int = 64) -> Float[Array, "output_dim input_dim"]:
    shape = (output_dim, input_dim)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 17) / 19


def _assert_close(result: Float[Array, "*shape"], reference: Float[Array, "*shape"]) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _hadamard_transform_output_axis(
    weights: Float[Array, "*components out_channels in_channels"],
) -> Float[Array, "*components out_channels in_channels"]:
    transformed = hadamard_transform(jnp.swapaxes(weights, -1, -2), block_size=32)
    return jnp.swapaxes(transformed, -1, -2)


def _process_preconditioner_block(
    block: Float[Array, "channels channels"],
    signs: Int[Array, " channels"],
) -> Float[Array, "channels channels"]:
    signed_block = block * signs[..., None] * signs[None, ...]
    transformed_input = hadamard_transform(signed_block, block_size=32)
    return _hadamard_transform_output_axis(transformed_input)


@dataclass(frozen=True)
class _PreconditionerRecordingSpec(WeightMatrixSpec):
    calls: list[Preconditioner | None]

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        is_sharded: bool = True,
    ) -> FullPrecisionMatrix:
        self.calls.append(preconditioner)
        return FullPrecisionSpec().compress(weights, implementation=implementation, is_sharded=is_sharded)


def test_hybrid_spec_roundtrips_json() -> None:
    spec = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=LowRankSpec(rank=2),
        incoherence_block_size=32,
        incoherence_processing_mode=IncoherenceProcessingMode.INPUT,
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
    input_signs = matrix.incoherence_signs.input_signs
    output_signs = matrix.incoherence_signs.output_signs
    assert input_signs is not None
    assert output_signs is not None
    assert input_signs.shape == (64,)
    assert output_signs.shape == (32,)
    _assert_close(result=matrix.decompress(), reference=weights)


def test_hybrid_input_incoherence_leaves_output_basis_unprocessed() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
        incoherence_processing_mode=IncoherenceProcessingMode.INPUT,
    ).compress(weights, key=jax.random.key(1))
    vector = (jnp.arange(64, dtype=jnp.float32) - 3) / 7

    assert matrix.incoherence_signs is not None
    input_signs = matrix.incoherence_signs.input_signs
    assert input_signs is not None
    assert matrix.incoherence_signs.output_signs is None
    _assert_close(
        result=matrix.quantized.decompress(),
        reference=hadamard_transform(weights * input_signs, block_size=32),
    )
    _assert_close(result=matrix.dot(vector, keychain=Keychain.init(0)), reference=weights @ vector)
    _assert_close(result=matrix.decompress(), reference=weights)


def test_hybrid_output_incoherence_leaves_input_basis_unprocessed() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
        incoherence_processing_mode=IncoherenceProcessingMode.OUTPUT,
    ).compress(weights, key=jax.random.key(2))
    vector = (jnp.arange(64, dtype=jnp.float32) - 3) / 7

    assert matrix.incoherence_signs is not None
    output_signs = matrix.incoherence_signs.output_signs
    assert matrix.incoherence_signs.input_signs is None
    assert output_signs is not None
    _assert_close(
        result=matrix.quantized.decompress(),
        reference=_hadamard_transform_output_axis(weights * output_signs[..., None]),
    )
    _assert_close(result=matrix.dot(vector, keychain=Keychain.init(0)), reference=weights @ vector)
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


def test_hybrid_transposed_dot_is_not_supported() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=FullPrecisionSpec(),
        adapter_spec=None,
        incoherence_block_size=32,
    ).compress(weights, key=jax.random.key(2))
    vector = (jnp.arange(32, dtype=jnp.float32) + 5) / 11

    with pytest.raises(ValueError, match="Transposed matmul is not supported"):
        matrix.dot(vector, keychain=Keychain.init(1), transposed=True)


def test_hybrid_adapter_compresses_residual_with_original_input_and_transformed_output_basis() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=LowRankSpec(rank=1),
        adapter_spec=FullPrecisionSpec(),
        incoherence_block_size=32,
    ).compress(weights, key=jax.random.key(3))

    assert matrix.incoherence_signs is not None
    assert matrix.adapter is not None
    transformed_weights = matrix.incoherence_signs.process_weights(weights, block_size=32)
    transformed_residual = transformed_weights - matrix.quantized.decompress()
    expected_adapter = matrix.incoherence_signs.unprocess_weight_input_axis(
        transformed_residual,
        block_size=32,
    )

    _assert_close(result=matrix.adapter.decompress(), reference=expected_adapter)
    _assert_close(result=matrix.decompress(), reference=weights)


def test_hybrid_adapter_dot_matches_original_weights() -> None:
    weights = _weights()
    matrix = HybridSpec(
        quantization_spec=LowRankSpec(rank=1),
        adapter_spec=FullPrecisionSpec(),
        incoherence_block_size=32,
    ).compress(weights, key=jax.random.key(4))
    vector = (jnp.arange(64, dtype=jnp.float32) - 13) / 17

    result = matrix.dot(vector, keychain=Keychain.init(2))

    _assert_close(result=result, reference=weights @ vector)


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


def test_hybrid_incoherence_adapts_preconditioner_to_adapter_basis() -> None:
    quantization_calls: list[Preconditioner | None] = []
    adapter_calls: list[Preconditioner | None] = []
    input_block = jnp.diag(jnp.linspace(1, 2, 64, dtype=jnp.float32))
    output_block = jnp.diag(jnp.linspace(3, 4, 32, dtype=jnp.float32))
    preconditioner = Preconditioner.init(input_block=input_block, output_block=output_block)
    spec = HybridSpec(
        quantization_spec=_PreconditionerRecordingSpec(quantization_calls),
        adapter_spec=_PreconditionerRecordingSpec(adapter_calls),
        incoherence_block_size=32,
    )

    matrix = spec.compress(_weights(), key=jax.random.key(6), preconditioner=preconditioner)

    assert matrix.incoherence_signs is not None
    assert len(quantization_calls) == 1
    assert len(adapter_calls) == 1
    quantization_preconditioner = quantization_calls[0]
    adapter_preconditioner = adapter_calls[0]
    assert quantization_preconditioner is not None
    assert adapter_preconditioner is not None
    quantization_input_block = quantization_preconditioner.input_block
    quantization_output_block = quantization_preconditioner.output_block
    adapter_input_block = adapter_preconditioner.input_block
    adapter_output_block = adapter_preconditioner.output_block
    assert quantization_input_block is not None
    assert quantization_output_block is not None
    assert adapter_input_block is not None
    assert adapter_output_block is not None
    input_signs = matrix.incoherence_signs.input_signs
    output_signs = matrix.incoherence_signs.output_signs
    assert input_signs is not None
    assert output_signs is not None
    _assert_close(
        result=quantization_input_block,
        reference=_process_preconditioner_block(input_block, input_signs),
    )
    _assert_close(
        result=quantization_output_block,
        reference=_process_preconditioner_block(output_block, output_signs),
    )
    _assert_close(result=adapter_input_block, reference=input_block)
    _assert_close(
        result=adapter_output_block,
        reference=_process_preconditioner_block(output_block, output_signs),
    )


def test_hybrid_input_incoherence_keeps_output_preconditioner_basis() -> None:
    quantization_calls: list[Preconditioner | None] = []
    input_block = jnp.diag(jnp.linspace(1, 2, 64, dtype=jnp.float32))
    output_block = jnp.diag(jnp.linspace(3, 4, 32, dtype=jnp.float32))
    preconditioner = Preconditioner.init(input_block=input_block, output_block=output_block)
    spec = HybridSpec(
        quantization_spec=_PreconditionerRecordingSpec(quantization_calls),
        adapter_spec=None,
        incoherence_block_size=32,
        incoherence_processing_mode=IncoherenceProcessingMode.INPUT,
    )

    matrix = spec.compress(_weights(), key=jax.random.key(7), preconditioner=preconditioner)

    assert matrix.incoherence_signs is not None
    input_signs = matrix.incoherence_signs.input_signs
    assert input_signs is not None
    assert matrix.incoherence_signs.output_signs is None
    assert len(quantization_calls) == 1
    quantization_preconditioner = quantization_calls[0]
    assert quantization_preconditioner is not None
    quantization_input_block = quantization_preconditioner.input_block
    quantization_output_block = quantization_preconditioner.output_block
    assert quantization_input_block is not None
    assert quantization_output_block is not None
    _assert_close(
        result=quantization_input_block,
        reference=_process_preconditioner_block(input_block, input_signs),
    )
    _assert_close(result=quantization_output_block, reference=output_block)
