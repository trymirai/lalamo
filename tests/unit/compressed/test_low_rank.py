from math import prod

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding

from lalamo.compressed.low_rank import LowRankMatrix, LowRankSpec
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    Layout,
    Preconditioner,
    WeightMatrixSpec,
)
from tests.common import assert_close

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 4, 4)
    weights = (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 5) / 7
    return jax.device_put(weights, _unsharded_sharding(len(shape)))


def _manual_low_rank_decompress(weights: jax.Array, rank: int) -> jax.Array:
    u, singular_values, vh = jnp.linalg.svd(weights, full_matrices=False)
    singular_value_roots = jnp.sqrt(singular_values[..., :rank])
    up_projection = u[..., :, :rank] * singular_value_roots[..., None, :]
    down_projection = singular_value_roots[..., :, None] * vh[..., :rank, :]
    return up_projection @ down_projection


def _positive_definite_block(seed: int) -> jax.Array:
    factors = jax.random.normal(jax.random.key(seed), (4, 4), dtype=jnp.float32)
    return factors @ factors.T / 4 + 0.5 * jnp.identity(4, dtype=jnp.float32)


def _apply_preconditioner(weights: jax.Array, preconditioner: Preconditioner) -> jax.Array:
    result = weights
    output_block = preconditioner.output_block
    if output_block is not None:
        output_factor = jnp.linalg.cholesky(output_block, upper=True)
        result = output_factor @ result

    input_block = preconditioner.input_block
    if input_block is not None:
        input_factor = jnp.linalg.cholesky(input_block, upper=True)
        result = result @ input_factor.T

    return result


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _unsharded_sharding(ndim: int) -> NamedSharding:
    sharding = make_sharding((None,) * ndim)
    assert sharding is not None
    return sharding


def test_low_rank_spec_roundtrips_json() -> None:
    spec = LowRankSpec(rank=2)

    restored = WeightMatrixSpec.from_json(spec.to_json())

    assert restored == spec


@pytest.mark.parametrize("implementation", list(CompressionImplementation))
def test_low_rank_compress_accepts_any_implementation(implementation: CompressionImplementation) -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights(), implementation=implementation)

    assert isinstance(matrix, LowRankMatrix)


@pytest.mark.parametrize("rank", [1, 3])
def test_low_rank_compress_and_decompress_match_truncated_svd(rank: int) -> None:
    weights = _logical_weights()

    matrix = LowRankSpec(rank=rank).compress(weights)

    assert matrix.shape == weights.shape
    assert matrix.up_projection.shape == (4, rank)
    assert matrix.down_projection.shape == (rank, 4)
    _assert_close(result=matrix.decompress(), reference=_manual_low_rank_decompress(weights, rank))


def test_batched_low_rank_compress_and_decompress_match_truncated_svd() -> None:
    weights = _logical_weights(2)

    matrix = LowRankSpec(rank=2).compress(weights)

    assert matrix.shape == weights.shape
    assert matrix.up_projection.shape == (2, 4, 2)
    assert matrix.down_projection.shape == (2, 2, 4)
    assert matrix.up_projection.sharding == _unsharded_sharding(3)
    assert matrix.down_projection.sharding == _unsharded_sharding(3)
    _assert_close(result=matrix.decompress(), reference=_manual_low_rank_decompress(weights, 2))


def test_low_rank_compress_with_identity_preconditioner_matches_truncated_svd() -> None:
    weights = _logical_weights()

    matrix = LowRankSpec(rank=2).compress(weights, preconditioner=Preconditioner.identity())

    _assert_close(result=matrix.decompress(), reference=_manual_low_rank_decompress(weights, 2))


@pytest.mark.parametrize("input_seed, output_seed", [(11, None), (None, 17), (11, 17)])
def test_low_rank_compress_with_preconditioner_matches_truncated_svd_in_weighted_space(
    input_seed: int | None,
    output_seed: int | None,
) -> None:
    weights = _logical_weights()
    input_block = None
    if input_seed is not None:
        input_block = _positive_definite_block(input_seed)
    output_block = None
    if output_seed is not None:
        output_block = _positive_definite_block(output_seed)
    preconditioner = Preconditioner.init(input_block=input_block, output_block=output_block)

    matrix = LowRankSpec(rank=2).compress(weights, preconditioner=preconditioner)

    transformed_result = _apply_preconditioner(matrix.decompress(), preconditioner)
    transformed_reference = _manual_low_rank_decompress(_apply_preconditioner(weights, preconditioner), 2)
    _assert_close(result=transformed_result, reference=transformed_reference)


def test_low_rank_compress_maps_batched_preconditioned_weights() -> None:
    weights = jnp.stack([_logical_weights(), _logical_weights() * 0.5])
    input_blocks = jnp.stack([_positive_definite_block(11), _positive_definite_block(13)])
    output_blocks = jnp.stack([_positive_definite_block(17), _positive_definite_block(19)])
    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)
    spec = LowRankSpec(rank=2)

    def compress_one(weights: jax.Array, preconditioner: Preconditioner) -> jax.Array:
        return spec.compress(weights, preconditioner=preconditioner).decompress()

    result = spec.compress(weights, preconditioner=preconditioner).decompress()
    reference = jax.vmap(compress_one)(weights, preconditioner)

    _assert_close(result=result, reference=reference)


def test_low_rank_compress_rejects_rank_larger_than_svd_rank() -> None:
    with pytest.raises(ValueError, match="SVD only produced rank"):
        LowRankSpec(rank=5).compress(_logical_weights())


def test_low_rank_shape_dtype_struct_compress_uses_factor_shapes_and_partition(fake_mesh: Mesh) -> None:
    matrix = LowRankSpec(rank=2).compress(dummy_array((4, 4), jnp.float32, _unsharded_sharding(2)))

    assert matrix.shape == (4, 4)
    assert matrix.dtype == jnp.float32
    assert matrix.up_projection.shape == (4, 2)
    assert matrix.down_projection.shape == (2, 4)
    assert matrix.up_projection.sharding == _unsharded_sharding(2)
    assert matrix.down_projection.sharding == _unsharded_sharding(2)
    _assert_named_sharding(matrix.up_projection.sharding, fake_mesh)
    _assert_named_sharding(matrix.down_projection.sharding, fake_mesh)


def test_low_rank_export_contains_projections_and_spec() -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())

    exported = matrix.export()

    assert set(exported.arrays) == {"up_projection", "down_projection"}
    assert exported.metadata == {"spec": matrix.spec.to_json()}
    _assert_close(result=exported.arrays["up_projection"], reference=matrix.up_projection)
    _assert_close(result=exported.arrays["down_projection"], reference=matrix.down_projection)


def test_low_rank_export_load_roundtrips_and_keeps_projections_unsharded(fake_mesh: Mesh) -> None:
    spec = LowRankSpec(rank=2)
    original = spec.compress(_logical_weights())
    template = spec.compress(dummy_array(_logical_weights().shape, jnp.float32, _unsharded_sharding(2)))

    restored = template.load_exported(original.export())

    assert restored.spec == spec
    _assert_close(result=restored.decompress(), reference=original.decompress())
    _assert_named_sharding(restored.up_projection.sharding, fake_mesh)
    _assert_named_sharding(restored.down_projection.sharding, fake_mesh)
    assert restored.up_projection.sharding == _unsharded_sharding(2)
    assert restored.down_projection.sharding == _unsharded_sharding(2)
    assert restored.up_projection.sharding == template.up_projection.sharding
    assert restored.down_projection.sharding == template.down_projection.sharding


def test_low_rank_load_rejects_spec_mismatch() -> None:
    original = LowRankSpec(rank=2).compress(_logical_weights())
    template = LowRankSpec(rank=1).compress(dummy_array((4, 4), jnp.float32, _unsharded_sharding(2)))

    with pytest.raises(ValueError, match="spec mismatch"):
        template.load_exported(original.export())


def test_low_rank_to_full_precision_preserves_output_input_layout_and_dot() -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())
    full_precision = matrix.to_full_precision()
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), _unsharded_sharding(1))

    assert isinstance(full_precision, FullPrecisionMatrix)
    assert full_precision.spec.layout == Layout.OUTPUT_INPUT
    assert full_precision.shape == matrix.shape
    _assert_close(result=full_precision.decompress(), reference=matrix.decompress())
    _assert_close(
        result=full_precision.dot(vector, keychain=Keychain.init(0)),
        reference=matrix.dot(vector, keychain=Keychain.init(1)),
    )


def test_low_rank_decompress_removes_factor_sharding() -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())

    result = matrix.decompress()

    assert result.sharding == _unsharded_sharding(2)


def test_low_rank_dot_matches_decompressed_weights_and_keeps_output_unsharded(fake_mesh: Mesh) -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), _unsharded_sharding(1))

    result = matrix.dot(vector, keychain=Keychain.init(2))

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _unsharded_sharding(1)
    _assert_close(result=result, reference=matrix.decompress() @ vector)


def test_low_rank_transposed_dot_matches_decompressed_weights_and_keeps_output_unsharded(fake_mesh: Mesh) -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), _unsharded_sharding(1))

    result = matrix.dot(vector, keychain=Keychain.init(3), transposed=True)

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _unsharded_sharding(1)
    _assert_close(result=result, reference=matrix.decompress().T @ vector)


def test_low_rank_dot_vmapped_over_inputs_matches_reference_and_keeps_output_unsharded(fake_mesh: Mesh) -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())
    vectors = jax.device_put(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        _unsharded_sharding(2),
    )

    result = jax.vmap(lambda vector: matrix.dot(vector, keychain=Keychain.init(4)))(vectors)
    reference = jnp.einsum("oi,bi->bo", matrix.decompress(), vectors)

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _unsharded_sharding(2)
    _assert_close(result=result, reference=reference)


def test_batched_low_rank_dot_requires_vmap_and_matches_expert_reference(fake_mesh: Mesh) -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights(2))
    vectors = jax.device_put(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        _unsharded_sharding(2),
    )

    with pytest.raises(ValueError, match="Use vmap"):
        matrix.dot(
            jax.device_put(jnp.arange(4, dtype=jnp.float32), _unsharded_sharding(1)),
            keychain=Keychain.init(5),
        )

    result = jax.vmap(lambda matrix_row, vector: matrix_row.dot(vector, keychain=Keychain.init(6)))(matrix, vectors)
    reference = jnp.einsum("eoi,ei->eo", matrix.decompress(), vectors)

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _unsharded_sharding(2)
    _assert_close(result=result, reference=reference)


def test_low_rank_dot_under_jit_keeps_output_unsharded(fake_mesh: Mesh) -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), _unsharded_sharding(1))

    result = jax.jit(lambda matrix, vector: matrix.dot(vector, keychain=Keychain.init(7)))(matrix, vector)

    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _unsharded_sharding(1)
    _assert_close(result=result, reference=matrix.decompress() @ vector)


def test_low_rank_dot_output_dtype_matches_input_dtype() -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights())
    vector = jax.device_put(jnp.arange(4, dtype=jnp.bfloat16), _unsharded_sharding(1))

    result = matrix.dot(vector, keychain=Keychain.init(8))

    assert result.dtype == vector.dtype
    _assert_close(result=result, reference=matrix.astype(vector.dtype).decompress() @ vector)
