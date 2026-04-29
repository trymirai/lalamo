from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding

from lalamo.compressed.awq import AWQSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.module import Keychain, ShardingAxis
from lalamo.modules.utils import call_vmapped
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, EmbeddingMatrix, Layout, WeightMatrixSpec
from tests.common import assert_close

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


COMPRESSED_MATRIX_CASES = (
    pytest.param(CompressedMatrixCase("awq", _awq_spec, weight_offset=7, weight_divisor=8), id="awq"),
    pytest.param(CompressedMatrixCase("mlx", _mlx_spec, weight_offset=3, weight_divisor=5), id="mlx"),
)


def logical_weights(case: CompressedMatrixCase, *leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - case.weight_offset) / case.weight_divisor


def expected_packed_shape(stored_shape: tuple[int, ...], bits: int) -> tuple[int, ...]:
    *leading_dims, cols = stored_shape
    packed_cols = cols if bits == 8 else cols // (8 // bits)
    return (*leading_dims, packed_cols)


def assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def assert_close_arrays(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def logical_dot_reference(layout: Layout, dequantized_weights: jax.Array, vector: jax.Array) -> jax.Array:
    if layout == Layout.INPUT_OUTPUT:
        return vector @ dequantized_weights
    return dequantized_weights @ vector


def embedding_sharding() -> NamedSharding:
    sharding = make_sharding((None,))
    assert sharding is not None
    return sharding


def batched_embedding_sharding() -> NamedSharding:
    sharding = make_sharding((ShardingAxis.EXPERT, None))
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
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))
    reference = logical_dot_reference(layout, training.decompress(), vector)

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
def test_compressed_matrix_training_and_inference_lookup_embedding_match_selected_row(
    case: CompressedMatrixCase,
) -> None:
    training, inference = _compress_pair(case, Layout.INPUT_OUTPUT)
    token_index = 2

    training_result = training.lookup_embedding(token_index, keychain=Keychain.init(2))
    inference_result = inference.lookup_embedding(token_index, keychain=Keychain.init(3))

    assert_close_arrays(result=training_result, reference=training.decompress()[token_index, :])
    assert_close_arrays(result=inference_result, reference=inference.decompress()[token_index, :])
    assert_close_arrays(result=training_result, reference=inference_result)


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

    assert_close_arrays(result=result, reference=matrix.decompress()[token_index, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == embedding_sharding()


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_lookup_embedding_accepts_jax_scalar_index(
    case: CompressedMatrixCase,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case),
        layout=Layout.INPUT_OUTPUT,
        implementation=implementation,
    )
    token_index = jnp.array(1, dtype=jnp.int32)

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(5))

    assert_close_arrays(result=result, reference=matrix.decompress()[token_index, :])


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_lookup_embedding_vmapped_over_tokens_preserves_token_and_feature_sharding(
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
    token_indices = jax.device_put(jnp.array([0, 2], dtype=jnp.int32), make_sharding((ShardingAxis.DATA,)))

    result = jax.vmap(lambda token_index: matrix.lookup_embedding(token_index, keychain=Keychain.init(5)))(
        token_indices,
    )

    reference_indices = jnp.asarray(jax.device_get(token_indices))
    assert_close_arrays(result=result, reference=matrix.decompress()[reference_indices, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_lookup_embedding_call_vmapped_over_tokens_matches_selected_rows_and_preserve_sharding(
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
    token_indices = jax.device_put(jnp.array([1, 3], dtype=jnp.int32), make_sharding((ShardingAxis.DATA,)))

    result = call_vmapped(
        matrix.lookup_embedding,
        token_indices,
        keychain=Keychain.init(5),
        added_sharding_axis=ShardingAxis.DATA,
    )

    reference_indices = jnp.asarray(jax.device_get(token_indices))
    assert_close_arrays(result=result, reference=matrix.decompress()[reference_indices, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_lookup_embedding_requires_input_output_layout(
    case: CompressedMatrixCase,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case),
        layout=Layout.OUTPUT_INPUT,
        implementation=implementation,
    )

    with pytest.raises(ValueError, match="Embedding lookup not supported"):
        matrix.lookup_embedding(0, keychain=Keychain.init(6))


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_batched_compressed_matrix_lookup_embedding_requires_vmap(
    case: CompressedMatrixCase,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case, 2),
        layout=Layout.INPUT_OUTPUT,
        implementation=implementation,
    )

    with pytest.raises(ValueError, match="Use vmap"):
        matrix.lookup_embedding(0, keychain=Keychain.init(7))


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_batched_compressed_matrix_lookup_embedding_vmapped_over_experts_matches_reference_and_preserves_sharding(
    fake_mesh: Mesh,
    case: CompressedMatrixCase,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case, 2),
        layout=Layout.INPUT_OUTPUT,
        implementation=implementation,
    )
    token_index = 2

    result = jax.vmap(lambda matrix_row: matrix_row.lookup_embedding(token_index, keychain=Keychain.init(8)))(matrix)

    assert_close_arrays(result=result, reference=matrix.decompress()[:, token_index, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == batched_embedding_sharding()


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_lookup_embedding_under_jit_matches_reference_and_preserves_sharding(
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
    token_index = jnp.array(3, dtype=jnp.int32)

    result = jax.jit(lambda matrix, index: matrix.lookup_embedding(index, keychain=Keychain.init(9)))(
        matrix,
        token_index,
    )

    assert_close_arrays(result=result, reference=matrix.decompress()[token_index, :])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == embedding_sharding()


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_dot_vmapped_over_inputs_matches_reference_and_preserves_input_sharding(
    fake_mesh: Mesh,
    case: CompressedMatrixCase,
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case),
        layout=layout,
        implementation=implementation,
    )
    vectors = jax.device_put(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        make_sharding((None, ShardingAxis.DATA)),
    )

    result = jax.vmap(lambda vector: matrix.dot(vector, keychain=Keychain.init(10)))(vectors)
    reference = jax.vmap(lambda vector: logical_dot_reference(layout, matrix.decompress(), vector))(vectors)

    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vectors.sharding
    assert_close_arrays(result=result, reference=reference)


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_batched_compressed_matrix_dot_requires_vmap_and_matches_expert_reference(
    fake_mesh: Mesh,
    case: CompressedMatrixCase,
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case, 2),
        layout=layout,
        implementation=implementation,
    )
    vectors = jax.device_put(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        make_sharding((ShardingAxis.EXPERT, ShardingAxis.DATA)),
    )

    with pytest.raises(ValueError, match="Use vmap"):
        matrix.dot(
            jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,))),
            keychain=Keychain.init(11),
        )

    result = jax.vmap(lambda matrix_row, vector: matrix_row.dot(vector, keychain=Keychain.init(12)))(matrix, vectors)
    reference = jax.vmap(lambda matrix_row, vector: logical_dot_reference(layout, matrix_row.decompress(), vector))(
        matrix,
        vectors,
    )

    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vectors.sharding
    assert_close_arrays(result=result, reference=reference)


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_dot_under_jit_preserves_input_sharding(
    fake_mesh: Mesh,
    case: CompressedMatrixCase,
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case),
        layout=layout,
        implementation=implementation,
    )
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))

    result = jax.jit(lambda matrix, vector: matrix.dot(vector, keychain=Keychain.init(13)))(matrix, vector)
    reference = logical_dot_reference(layout, matrix.decompress(), vector)

    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vector.sharding
    assert_close_arrays(result=result, reference=reference)


@pytest.mark.parametrize("case", COMPRESSED_MATRIX_CASES)
@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_compressed_matrix_dot_works_with_single_device_input_sharding(
    case: CompressedMatrixCase,
    layout: Layout,
    implementation: CompressionImplementation,
) -> None:
    matrix = _compress(
        case,
        logical_weights(case),
        layout=layout,
        implementation=implementation,
    )
    vector = jnp.arange(4, dtype=jnp.float32)

    result = matrix.dot(vector, keychain=Keychain.init(14))
    reference = logical_dot_reference(layout, matrix.decompress(), vector)

    assert_close_arrays(result=result, reference=reference)
