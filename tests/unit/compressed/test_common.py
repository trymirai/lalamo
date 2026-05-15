from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding

from lalamo.compressed.e8p import E8PSpec
from lalamo.compressed.int import IntSpec
from lalamo.compressed.microfloat import MicrofloatSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.compressed.quantile import QuantileSpec
from lalamo.module import Keychain, ShardingAxis
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionSpec,
    Layout,
    WeightMatrixSpec,
)
from tests.common import assert_close_arrays, assert_named_sharding

pytestmark = pytest.mark.usefixtures("fake_mesh")


@dataclass(frozen=True)
class CompressedMatrixCase:
    name: str
    make_spec: Callable[[Literal[4, 8], int, Layout], WeightMatrixSpec]
    weight_offset: float
    weight_divisor: float


def _int_spec(bits: Literal[4, 8], group_size: int, layout: Layout) -> WeightMatrixSpec:
    return IntSpec(bits=bits, group_size=group_size, layout=layout)


def _mlx_spec(bits: Literal[4, 8], group_size: int, layout: Layout) -> WeightMatrixSpec:
    return MLXSpec(bits=bits, group_size=group_size, layout=layout)


def _quantile_spec(bits: Literal[4, 8], group_size: int, layout: Layout) -> WeightMatrixSpec:
    return QuantileSpec(bits=bits, group_size=group_size, layout=layout)


def _microfloat_spec(bits: Literal[4, 8], group_size: int, layout: Layout) -> WeightMatrixSpec:
    if bits != 4:
        raise ValueError(f"Microfloat only supports 4-bit weights, got {bits}")
    return MicrofloatSpec(group_size=group_size, layout=layout)


def _e8p_spec(bits: Literal[4, 8], group_size: int, layout: Layout) -> WeightMatrixSpec:  # noqa: ARG001
    if bits != 4:
        raise ValueError(f"E8P shared tests only use the 4-bit variant, got {bits}")
    return E8PSpec(bits=bits, layout=layout)


COMPRESSED_MATRIX_CASES = (
    pytest.param(CompressedMatrixCase("int", _int_spec, weight_offset=7, weight_divisor=8), id="int"),
    pytest.param(CompressedMatrixCase("mlx", _mlx_spec, weight_offset=3, weight_divisor=5), id="mlx"),
    pytest.param(
        CompressedMatrixCase("quantile", _quantile_spec, weight_offset=5, weight_divisor=6),
        id="quantile",
    ),
    pytest.param(
        CompressedMatrixCase("microfloat", _microfloat_spec, weight_offset=4, weight_divisor=4),
        id="microfloat",
    ),
    pytest.param(CompressedMatrixCase("e8p", _e8p_spec, weight_offset=11, weight_divisor=9), id="e8p"),
)


def logical_weights(case: CompressedMatrixCase, *leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 8, 8)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - case.weight_offset) / case.weight_divisor


def expected_packed_shape(stored_shape: tuple[int, ...], bits: int) -> tuple[int, ...]:
    *leading_dims, cols = stored_shape
    packed_cols = cols if bits == 8 else cols // (8 // bits)
    return (*leading_dims, packed_cols)


def host_array(array: jax.Array) -> jax.Array:
    return jnp.asarray(jax.device_get(array))


def host_decompressed(matrix: EmbeddingMatrix[WeightMatrixSpec]) -> jax.Array:
    return host_array(matrix.decompress())


def host_embedding_table(matrix: EmbeddingMatrix[WeightMatrixSpec]) -> jax.Array:
    match matrix.spec:
        case IntSpec() | E8PSpec() | MLXSpec() | QuantileSpec() | MicrofloatSpec() | FullPrecisionSpec() as spec:
            return host_array(spec.layout.from_output_input(matrix.decompress()))
    raise TypeError(f"Unsupported matrix spec type: {type(matrix.spec).__name__}")


def logical_dot_reference(dequantized_weights: jax.Array, vector: jax.Array) -> jax.Array:
    dequantized_weights = host_array(dequantized_weights)
    vector = host_array(vector)
    return dequantized_weights @ vector


def embedding_sharding() -> NamedSharding:
    sharding = make_sharding((None,))
    assert sharding is not None
    return sharding


def _compress(
    case: CompressedMatrixCase,
    weights: jax.Array,
    *,
    layout: Layout,
    implementation: CompressionImplementation,
    is_sharded: bool = True,
) -> EmbeddingMatrix[WeightMatrixSpec]:
    matrix = case.make_spec(4, 2, layout).compress(
        weights,
        implementation=implementation,
        is_sharded=is_sharded,
    )
    assert isinstance(matrix, EmbeddingMatrix)
    return matrix


def _compress_pair(
    case: CompressedMatrixCase,
    layout: Layout,
) -> tuple[EmbeddingMatrix[WeightMatrixSpec], EmbeddingMatrix[WeightMatrixSpec]]:
    weights = logical_weights(case)
    training = _compress(
        case,
        weights,
        layout=layout,
        implementation=CompressionImplementation.TRAINING,
    )
    inference = _compress(
        case,
        weights,
        layout=layout,
        implementation=CompressionImplementation.INFERENCE,
    )
    return training, inference


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_compressed_matrix_training_and_inference_outputs_match_reference_and_preserve_input_sharding(
    fake_mesh: Mesh,
    case: CompressedMatrixCase,
    layout: Layout,
) -> None:
    training, inference = _compress_pair(case, layout)
    vector = jax.device_put(jnp.arange(8, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))
    reference = logical_dot_reference(training.decompress(), vector)

    training_result = training.dot(vector, keychain=Keychain.init(0))
    inference_result = inference.dot(vector, keychain=Keychain.init(1))

    assert_close_arrays(result=training.decompress(), reference=inference.decompress())
    assert_close_arrays(result=training_result, reference=reference)
    assert_close_arrays(result=inference_result, reference=reference)
    assert_close_arrays(result=training_result, reference=inference_result)
    assert_named_sharding(training_result.sharding, fake_mesh)
    assert_named_sharding(inference_result.sharding, fake_mesh)
    assert training_result.sharding == vector.sharding
    assert inference_result.sharding == vector.sharding


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_lookup_embedding_matches_selected_row_and_feature_sharding(
    fake_mesh: Mesh,
    case: CompressedMatrixCase,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case),
        layout=Layout.INPUT_OUTPUT,
        implementation=implementation,
        is_sharded=False,
    )
    token_index = 2

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(4))

    assert_close_arrays(result=result, reference=host_embedding_table(matrix)[token_index, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == embedding_sharding()
