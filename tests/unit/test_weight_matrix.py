from math import prod

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, ShardingAxis
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, Layout, Preconditioner, WeightMatrixSpec
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


def _embedding_sharding() -> NamedSharding:
    sharding = make_sharding((ShardingAxis.TENSOR,))
    assert sharding is not None
    return sharding


def _batched_embedding_sharding() -> NamedSharding:
    sharding = make_sharding((ShardingAxis.EXPERT, ShardingAxis.TENSOR))
    assert sharding is not None
    return sharding


def test_preconditioner_preserves_symmetric_blocks() -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 3.0, 4.0],
            [0.5, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [7.0, 2.0],
            [2.0, 11.0],
        ],
        dtype=jnp.float32,
    )

    preconditioner = Preconditioner.init(input_block=input_block, output_block=output_block)

    assert preconditioner.input_block_tril is not None
    assert preconditioner.input_block_tril.shape == (6,)
    assert preconditioner.output_block_tril is not None
    assert preconditioner.output_block_tril.shape == (3,)
    assert preconditioner.input_block is not None
    assert preconditioner.output_block is not None
    _assert_close(result=preconditioner.input_block, reference=input_block)
    _assert_close(result=preconditioner.output_block, reference=output_block)


def test_preconditioner_identity_has_no_blocks() -> None:
    preconditioner = Preconditioner.identity()

    assert preconditioner.input_block is None
    assert preconditioner.output_block is None


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


def test_full_precision_lookup_embedding_matches_selected_row_and_feature_sharding(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = 2

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(5))

    _assert_close(result=result, reference=matrix.decompress()[token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _embedding_sharding()


def test_full_precision_lookup_embedding_accepts_jax_scalar_index(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = jnp.array(1, dtype=jnp.int32)

    result = matrix.lookup_embedding(token_index, keychain=Keychain.init(6))

    _assert_close(result=result, reference=matrix.decompress()[token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)


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

    _assert_close(result=result, reference=matrix.decompress()[:, token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _batched_embedding_sharding()


def test_full_precision_lookup_embedding_under_jit_matches_reference_and_preserves_sharding(fake_mesh: Mesh) -> None:
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_logical_weights())
    token_index = jnp.array(3, dtype=jnp.int32)

    result = jax.jit(lambda matrix, index: matrix.lookup_embedding(index, keychain=Keychain.init(10)))(
        matrix,
        token_index,
    )

    _assert_close(result=result, reference=matrix.decompress()[token_index, :])
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _embedding_sharding()


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_full_precision_init_uses_layout_shape_and_partition(fake_mesh: Mesh, layout: Layout) -> None:
    matrix = FullPrecisionSpec(layout=layout).init(
        EmptyInitializer(dtype=jnp.float32),
        leading_dims=(2,),
        output_dim=4,
        input_dim=4,
    )
    expected_sharding = make_sharding(layout.weight_partition(num_leading_dims=1))

    _assert_named_sharding(matrix.weights.sharding, fake_mesh)
    assert matrix.weights.shape == layout.weight_shape((2,), output_dim=4, input_dim=4)
    assert matrix.weights.sharding == expected_sharding
