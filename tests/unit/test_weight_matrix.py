from math import prod

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding

from lalamo.compressed.awq import AWQSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.initializer import EmptyInitializer, RandomInitializer
from lalamo.module import Keychain, ShardingAxis
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import (
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    ShapeDtypeMatrix,
    WeightMatrixSpec,
)
from tests.common import assert_close


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    return jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) / 10


def _stored_weights(layout: Layout, weights: jax.Array) -> jax.Array:
    if layout == Layout.INPUT_OUTPUT:
        return weights.swapaxes(-1, -2)
    return weights


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _ragged_dot_reference(matrix: FullPrecisionMatrix, vectors: jax.Array, group_sizes: jax.Array) -> jax.Array:
    weights = _host_decompressed(matrix)
    host_group_sizes = jnp.asarray(jax.device_get(group_sizes))
    expert_indices = jnp.repeat(
        jnp.arange(weights.shape[0]),
        host_group_sizes,
        total_repeat_length=vectors.shape[0],
    )
    return jnp.einsum("toi,ti->to", weights[expert_indices], jnp.asarray(jax.device_get(vectors)))


def _host_decompressed(matrix: FullPrecisionMatrix) -> jax.Array:
    return jnp.asarray(jax.device_get(matrix.decompress()))


def _host_embedding_table(matrix: FullPrecisionMatrix) -> jax.Array:
    return jnp.asarray(jax.device_get(matrix.weights))


def _embedding_sharding() -> NamedSharding:
    sharding = make_sharding((None,))
    assert sharding is not None
    return sharding


def _batched_embedding_sharding() -> NamedSharding:
    sharding = make_sharding((ShardingAxis.EXPERT, None))
    assert sharding is not None
    return sharding


def _weight_sharding(layout: Layout) -> NamedSharding:
    sharding = make_sharding(layout.weight_partition(num_leading_dims=0))
    assert sharding is not None
    return sharding


def _unsharded_weights() -> NamedSharding:
    sharding = make_sharding((None, None))
    assert sharding is not None
    return sharding


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_weight_matrix_numel_returns_product_of_shape(layout: Layout) -> None:
    weights = jnp.ones((2, 3, 5), dtype=jnp.float32)

    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    assert matrix.numel == 30


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_spec_roundtrips_json(layout: Layout) -> None:
    spec = FullPrecisionSpec(layout=layout)

    restored = WeightMatrixSpec.from_json(spec.to_json())

    assert restored == spec


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_export_load_roundtrips_weights_spec_and_template_sharding(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights()
    spec = FullPrecisionSpec(layout=layout)
    saved_sharding = make_sharding((None, None))
    original = FullPrecisionMatrix(
        spec=spec,
        weights=jax.device_put(_stored_weights(layout, weights), saved_sharding),
    )
    skeleton = spec.compress(dummy_array(weights.shape, weights.dtype))

    restored = skeleton.load_exported(original.export())

    _assert_named_sharding(skeleton.weights.sharding, fake_mesh)
    assert original.export().metadata["spec"] == spec.to_json()
    assert restored.spec == spec
    assert restored.weights.sharding == skeleton.weights.sharding
    assert restored.weights.sharding != original.weights.sharding
    _assert_close(result=restored.weights, reference=original.weights)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_compress_overrides_input_sharding(fake_mesh: Mesh, layout: Layout) -> None:
    weights = jax.device_put(_logical_weights(), make_sharding((ShardingAxis.DATA, None)))

    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    assert matrix.weights.sharding == _weight_sharding(layout)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_decompress_removes_weight_sharding(fake_mesh: Mesh, layout: Layout) -> None:
    matrix = FullPrecisionSpec(layout=layout).compress(_logical_weights())

    result = matrix.decompress()

    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    assert result.sharding == _unsharded_weights()


def test_full_precision_load_rejects_spec_mismatch(fake_mesh: Mesh) -> None:
    weights = _logical_weights()
    original = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(weights)
    skeleton = FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(weights)

    _assert_named_sharding(skeleton.weights.sharding, fake_mesh)
    with pytest.raises(ValueError, match="spec mismatch"):
        skeleton.load_exported(original.export())


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_dot_matches_logical_matmul_and_preserves_input_sharding(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights()
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))
    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    result = matrix.dot(vector, keychain=Keychain.init(0))

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vector.sharding
    _assert_close(result=result, reference=weights @ vector)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_dot_vmapped_over_inputs_matches_reference_and_preserves_input_sharding(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights()
    matrix = FullPrecisionSpec(layout=layout).compress(weights)
    vectors = jax.device_put(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        make_sharding((None, ShardingAxis.DATA)),
    )

    result = jax.vmap(lambda vector: matrix.dot(vector, keychain=Keychain.init(1)))(vectors)

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vectors.sharding
    _assert_close(result=result, reference=jnp.einsum("oi,bi->bo", weights, vectors))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_batched_full_precision_dot_requires_vmap_and_matches_expert_reference(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights(2)
    matrix = FullPrecisionSpec(layout=layout).compress(weights)
    vectors = jax.device_put(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        make_sharding((ShardingAxis.EXPERT, ShardingAxis.DATA)),
    )

    with pytest.raises(ValueError, match="Use vmap"):
        matrix.dot(
            jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,))),
            keychain=Keychain.init(2),
        )

    result = jax.vmap(lambda matrix_row, vector: matrix_row.dot(vector, keychain=Keychain.init(3)))(matrix, vectors)

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vectors.sharding
    _assert_close(result=result, reference=jnp.einsum("eoi,ei->eo", weights, vectors))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_dot_under_jit_preserves_input_sharding(fake_mesh: Mesh, layout: Layout) -> None:
    weights = _logical_weights()
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))
    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    result = jax.jit(lambda matrix, vector: matrix.dot(vector, keychain=Keychain.init(4)))(matrix, vector)

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vector.sharding
    _assert_close(result=result, reference=weights @ vector)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_dot_works_with_single_device_input_sharding(layout: Layout) -> None:
    weights = _logical_weights()
    vector = jnp.arange(4, dtype=jnp.float32)
    matrix = FullPrecisionMatrix(
        spec=FullPrecisionSpec(layout=layout),
        weights=_stored_weights(layout, weights),
    )

    result = matrix.dot(vector, keychain=Keychain.init(4))

    _assert_close(result=result, reference=weights @ vector)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_dot_output_dtype_matches_input_dtype(fake_mesh: Mesh, layout: Layout) -> None:
    weights = _logical_weights()
    vector = jnp.arange(4, dtype=jnp.bfloat16)
    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    result = matrix.dot(vector, keychain=Keychain.init(15))

    assert result.dtype == vector.dtype
    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    _assert_close(result=result, reference=weights.astype(vector.dtype) @ vector)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_batched_full_precision_ragged_dot_matches_reference_and_preserves_input_sharding(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights(4)
    vectors = jax.device_put(
        jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10,
        make_sharding((ShardingAxis.DATA, None)),
    )
    group_sizes = jnp.array([2, 0, 0, 2], dtype=jnp.int32)
    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    result = matrix.ragged_dot(vectors, group_sizes, keychain=Keychain.init(16))

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vectors.sharding
    _assert_close(result=result, reference=_ragged_dot_reference(matrix, vectors, group_sizes))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_batched_full_precision_ragged_dot_under_jit_preserves_input_sharding(
    fake_mesh: Mesh,
    layout: Layout,
) -> None:
    weights = _logical_weights(2)
    vectors = jax.device_put(
        jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10,
        make_sharding((ShardingAxis.DATA, None)),
    )
    group_sizes = jnp.array([2, 2], dtype=jnp.int32)
    matrix = FullPrecisionSpec(layout=layout).compress(weights)

    result = jax.jit(lambda matrix, vectors: matrix.ragged_dot(vectors, group_sizes, keychain=Keychain.init(17)))(
        matrix,
        vectors,
    )

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == vectors.sharding
    _assert_close(result=result, reference=_ragged_dot_reference(matrix, vectors, group_sizes))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_batched_full_precision_ragged_dot_output_dtype_matches_input_dtype(layout: Layout) -> None:
    matrix = FullPrecisionSpec(layout=layout).compress(_logical_weights(2))
    vectors = jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4)
    group_sizes = jnp.array([1, 2], dtype=jnp.int32)

    result = matrix.ragged_dot(vectors, group_sizes, keychain=Keychain.init(18))

    assert result.dtype == vectors.dtype
    _assert_close(result=result, reference=_ragged_dot_reference(matrix.astype(vectors.dtype), vectors, group_sizes))


def test_full_precision_lookup_embedding_matches_selected_row_and_feature_sharding(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = 2

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(5))

    _assert_close(result=result, reference=_host_embedding_table(matrix)[token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _embedding_sharding()


def test_full_precision_lookup_embedding_uses_requested_dtype(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = 2

    result = matrix.lookup_embedding(token_index, dtype=jnp.bfloat16, keychain=Keychain.init(16))

    assert result.dtype == jnp.bfloat16
    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    _assert_close(result=result, reference=_host_embedding_table(matrix)[token_index, :].astype(jnp.bfloat16))


def test_full_precision_lookup_embedding_accepts_jax_scalar_index(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = jnp.array(1, dtype=jnp.int32)

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(6))

    _assert_close(result=result, reference=_host_embedding_table(matrix)[token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)


def test_full_precision_lookup_embedding_vmapped_over_tokens_keeps_token_and_feature_axes_unsharded(
    fake_mesh: Mesh,
) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_indices = jnp.array([0, 2], dtype=jnp.int32)

    result = jax.vmap(lambda token_index: matrix.lookup_embedding(token_index, keychain=Keychain.init(6)))(
        token_indices,
    )

    reference_indices = jnp.asarray(jax.device_get(token_indices))
    _assert_close(result=result, reference=_host_embedding_table(matrix)[reference_indices, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None, None))


def test_full_precision_lookup_embedding_call_vmapped_over_tokens_matches_selected_rows_and_keeps_axes_unsharded(
    fake_mesh: Mesh,
) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_indices = jnp.array([1, 3], dtype=jnp.int32)

    result = call_vmapped(
        matrix.lookup_embedding,
        token_indices,
        keychain=Keychain.init(6),
    )

    reference_indices = jnp.asarray(jax.device_get(token_indices))
    _assert_close(result=result, reference=_host_embedding_table(matrix)[reference_indices, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None, None))


def test_full_precision_lookup_embedding_rejects_output_input_layout(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(_logical_weights())

    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    with pytest.raises(ValueError, match="Embedding lookup not supported"):
        matrix.lookup_embedding(0, keychain=Keychain.init(7))


def test_batched_full_precision_lookup_embedding_requires_vmap(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights(2))

    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    with pytest.raises(ValueError, match="Use vmap"):
        matrix.lookup_embedding(0, keychain=Keychain.init(8))


def test_batched_full_precision_lookup_embedding_vmapped_over_experts_matches_reference_and_preserves_sharding(
    fake_mesh: Mesh,
) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights(2))
    token_index = 2

    result = jax.vmap(lambda matrix_row: matrix_row.lookup_embedding(token_index, keychain=Keychain.init(9)))(matrix)

    _assert_close(result=result, reference=_host_embedding_table(matrix)[:, token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _batched_embedding_sharding()


def test_full_precision_lookup_embedding_under_jit_matches_reference_and_preserves_sharding(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = jnp.array(3, dtype=jnp.int32)

    result = jax.jit(lambda matrix, index: matrix.lookup_embedding(index, keychain=Keychain.init(10)))(
        matrix,
        token_index,
    )

    _assert_close(result=result, reference=_host_embedding_table(matrix)[token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _embedding_sharding()


def test_empty_initializer_weight_matrix_returns_shape_dtype_matrix() -> None:
    matrix = EmptyInitializer(dtype=jnp.float32).weight_matrix(
        output_dim=4,
        input_dim=3,
        mixture_size=2,
    )

    assert isinstance(matrix, ShapeDtypeMatrix)
    assert matrix.shape == (2, 4, 3)
    assert matrix.dtype == jnp.float32
    assert matrix.decompress().shape == (2, 4, 3)


def test_random_initializer_embedding_matrix_uses_model_vocab_logical_shape(fake_mesh: Mesh) -> None:
    matrix = RandomInitializer(dtype=jnp.float32, key=jax.random.key(0)).embedding_matrix(
        vocabulary_size=5,
        model_dim=4,
    )

    assert isinstance(matrix, FullPrecisionMatrix)
    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    assert matrix.shape == (5, 4)
    assert matrix.decompress().shape == (4, 5)


def test_shape_dtype_embedding_load_exported_uses_logical_model_vocab_shape(fake_mesh: Mesh) -> None:
    weights = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)
    original = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(weights)
    template = EmptyInitializer(dtype=jnp.float32).embedding_matrix(vocabulary_size=5, model_dim=4)

    restored = template.load_exported(original.export())

    assert isinstance(template, ShapeDtypeMatrix)
    assert template.shape == (5, 4)
    assert template.decompress().shape == (4, 5)
    assert isinstance(restored, FullPrecisionMatrix)
    assert restored.shape == (5, 4)
    _assert_named_sharding(restored.weights.sharding, fake_mesh)
    _assert_close(
        result=restored.lookup_embedding(2, keychain=Keychain.init(11)),
        reference=original.lookup_embedding(2, keychain=Keychain.init(12)),
    )


@pytest.mark.parametrize(
    "spec",
    [
        FullPrecisionSpec(),
        AWQSpec(bits=8, group_size=2),
        MLXSpec(bits=8, group_size=2),
    ],
)
def test_shape_dtype_matrix_load_exported_uses_saved_weight_matrix_spec(spec: WeightMatrixSpec) -> None:
    original = spec.compress(_logical_weights())
    template = EmptyInitializer(dtype=jnp.float32).weight_matrix(output_dim=4, input_dim=4)

    restored = template.load_exported(original.export())

    assert not isinstance(restored, ShapeDtypeMatrix)
    assert restored.spec == spec
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype
    _assert_close(result=restored.decompress(), reference=original.decompress())
