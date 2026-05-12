from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding

from lalamo.compressed.awq import AWQSpec
from lalamo.compressed.e8p import E8PSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.compressed.mxfp4 import MXFP4Spec
from lalamo.compressed.normal_float import NormalFloatSpec
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

type Bits = Literal[4, 8]

pytestmark = pytest.mark.usefixtures("fake_mesh")


@dataclass(frozen=True)
class CompressedMatrixCase:
    name: str
    make_spec: Callable[[Bits, int, Layout], WeightMatrixSpec]
    weight_offset: float
    weight_divisor: float


def _awq_spec(bits: Bits, group_size: int, layout: Layout) -> WeightMatrixSpec:
    return AWQSpec(bits=bits, group_size=group_size, layout=layout)


def _mlx_spec(bits: Bits, group_size: int, layout: Layout) -> WeightMatrixSpec:
    return MLXSpec(bits=bits, group_size=group_size, layout=layout)


def _normal_float_spec(bits: Bits, group_size: int, layout: Layout) -> WeightMatrixSpec:
    return NormalFloatSpec(bits=bits, group_size=group_size, layout=layout)


def _mxfp4_spec(bits: Bits, group_size: int, layout: Layout) -> WeightMatrixSpec:
    if bits != 4:
        raise ValueError(f"MXFP4 only supports 4-bit weights, got {bits}")
    return MXFP4Spec(group_size=group_size, layout=layout)


def _e8p_spec(bits: Bits, group_size: int, layout: Layout) -> WeightMatrixSpec:  # noqa: ARG001
    if bits != 4:
        raise ValueError(f"E8P shared tests only use the 4-bit variant, got {bits}")
    return E8PSpec(bits=bits, layout=layout)


COMPRESSED_MATRIX_CASES = (
    pytest.param(CompressedMatrixCase("awq", _awq_spec, weight_offset=7, weight_divisor=8), id="awq"),
    pytest.param(CompressedMatrixCase("mlx", _mlx_spec, weight_offset=3, weight_divisor=5), id="mlx"),
    pytest.param(
        CompressedMatrixCase("normal_float", _normal_float_spec, weight_offset=5, weight_divisor=6),
        id="normal-float",
    ),
    pytest.param(CompressedMatrixCase("mxfp4", _mxfp4_spec, weight_offset=4, weight_divisor=4), id="mxfp4"),
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
        case AWQSpec() | E8PSpec() | MLXSpec() | NormalFloatSpec() | MXFP4Spec() | FullPrecisionSpec() as spec:
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
) -> EmbeddingMatrix[WeightMatrixSpec]:
    matrix = case.make_spec(4, 2, layout).compress(weights, implementation=implementation)
    return cast("EmbeddingMatrix[WeightMatrixSpec]", matrix)


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
    )
    token_index = 2

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(4))

    assert_close_arrays(result=result, reference=host_embedding_table(matrix)[token_index, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == embedding_sharding()
