from math import prod

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding

from lalamo.compressed.low_rank import LowRankSpec
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from tests.common import assert_close_arrays, assert_named_sharding
from tests.helpers import make_sharding, make_test_sharding_config

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


def _unsharded_sharding(ndim: int) -> NamedSharding:
    sharding = make_sharding((None,) * ndim)
    assert sharding is not None
    return sharding


def test_low_rank_compress_and_decompress_match_truncated_svd() -> None:
    weights = _logical_weights()
    rank = 2

    matrix = LowRankSpec(rank=rank).compress(weights, sharding_config=make_test_sharding_config())

    assert_close_arrays(result=matrix.decompress(), reference=_manual_low_rank_decompress(weights, rank))


def test_low_rank_compress_with_preconditioner_matches_truncated_svd_in_weighted_space() -> None:
    weights = _logical_weights()
    preconditioner = Preconditioner.init(
        input_block=_positive_definite_block(11),
        output_block=_positive_definite_block(17),
    )

    matrix = LowRankSpec(rank=2).compress(
        weights, preconditioner=preconditioner, sharding_config=make_test_sharding_config()
    )

    transformed_result = _apply_preconditioner(matrix.decompress(), preconditioner)
    transformed_reference = _manual_low_rank_decompress(_apply_preconditioner(weights, preconditioner), 2)
    assert_close_arrays(result=transformed_result, reference=transformed_reference)


def test_low_rank_compress_rejects_rank_larger_than_svd_rank() -> None:
    with pytest.raises(ValueError, match="SVD only produced rank"):
        LowRankSpec(rank=5).compress(_logical_weights(), sharding_config=make_test_sharding_config())


def test_low_rank_dot_matches_decompressed_weights_and_keeps_output_unsharded(fake_mesh: Mesh) -> None:
    matrix = LowRankSpec(rank=2).compress(_logical_weights(), sharding_config=make_test_sharding_config())
    vector = jax.device_put(jnp.arange(4, dtype=jnp.float32), _unsharded_sharding(1))

    result = matrix.dot(vector, keychain=Keychain.init(2, sharding_config=make_test_sharding_config()))

    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == _unsharded_sharding(1)
    assert_close_arrays(result=result, reference=matrix.decompress() @ vector)
