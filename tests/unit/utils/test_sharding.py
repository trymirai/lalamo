import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, typeof
from jax.sharding import Mesh, NamedSharding

from lalamo.module import ShardingAxis
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import is_sharded, make_sharding, reshard_as, sharding_of, use_out_sharding, with_sharding


def test_is_sharded_accepts_non_empty_named_sharding(fake_mesh: Mesh) -> None:
    sharding = make_sharding((ShardingAxis.DATA,))

    assert is_sharded(sharding)
    assert sharding.mesh == fake_mesh


def test_is_sharded_rejects_none_and_empty_mesh_sharding() -> None:
    assert not is_sharded(None)
    assert not is_sharded(typeof(jnp.arange(4, dtype=jnp.float32)).sharding)


def test_sharding_of_returns_concrete_array_sharding(fake_mesh: Mesh) -> None:
    array = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))

    sharding = sharding_of(array)

    assert isinstance(sharding, NamedSharding)
    assert sharding == array.sharding
    assert sharding.mesh == fake_mesh


def test_with_sharding_updates_shape_dtype_struct_sharding(fake_mesh: Mesh) -> None:
    source = dummy_array((4,), jnp.float32)
    sharding = make_sharding((ShardingAxis.DATA,))

    result = with_sharding(source, sharding)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == sharding
    assert result.sharding.mesh == fake_mesh


def test_reshard_as_uses_reference_named_sharding(fake_mesh: Mesh) -> None:
    source = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.TENSOR,)))
    reference = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((ShardingAxis.DATA,)))

    result = reshard_as(source, reference)

    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == reference.sharding
    assert result.sharding.mesh == fake_mesh
    np.testing.assert_array_equal(jax.device_get(result), jax.device_get(source))


def test_reshard_as_leaves_arrays_without_named_mesh_sharding_unchanged() -> None:
    source = jnp.arange(4, dtype=jnp.float32)
    reference = jnp.arange(4, dtype=jnp.float32)

    result = reshard_as(source, reference)

    assert result is source
    assert jnp.array_equal(result, source)


def test_reshard_as_updates_shape_dtype_struct_to_reference_sharding(fake_mesh: Mesh) -> None:
    source = dummy_array((4,), jnp.float32, make_sharding((ShardingAxis.TENSOR,)))
    reference = dummy_array((4,), jnp.float32, make_sharding((ShardingAxis.DATA,)))

    result = reshard_as(source, reference)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == reference.sharding
    assert result.sharding.mesh == fake_mesh
    assert result.shape == source.shape
    assert result.dtype == source.dtype


def test_reshard_as_places_array_on_shape_dtype_struct_reference_sharding(fake_mesh: Mesh) -> None:
    source = jnp.arange(4, dtype=jnp.float32)
    reference = dummy_array((4,), jnp.float32, make_sharding((ShardingAxis.DATA,)))

    result = reshard_as(source, reference)

    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == reference.sharding
    assert result.sharding.mesh == fake_mesh
    np.testing.assert_array_equal(jax.device_get(result), jax.device_get(source))


def test_reshard_as_clears_shape_dtype_struct_sharding_when_reference_has_none(fake_mesh: Mesh) -> None:
    del fake_mesh
    source = dummy_array((4,), jnp.float32, make_sharding((ShardingAxis.TENSOR,)))
    reference = dummy_array((4,), jnp.float32)

    result = reshard_as(source, reference)

    assert isinstance(result, ShapeDtypeStruct)
    assert result.sharding is None


def test_use_out_sharding_resolves_vmapped_scalar_gather(fake_mesh: Mesh) -> None:
    table = jax.device_put(
        jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
        make_sharding((None, ShardingAxis.TENSOR)),
    )
    indices = jax.device_put(jnp.array([0, 2], dtype=jnp.int32), make_sharding((ShardingAxis.DATA,)))

    @use_out_sharding((None,))
    def lookup_row(table: jax.Array, index: jax.Array) -> jax.Array:
        return table[index, :]

    result = jax.vmap(lambda index: lookup_row(table, index))(indices)

    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))
    assert result.sharding.mesh == fake_mesh
    np.testing.assert_array_equal(jax.device_get(result), np.asarray([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=np.float32))
