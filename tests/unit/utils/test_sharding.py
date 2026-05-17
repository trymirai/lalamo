import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import ShapeDtypeStruct, typeof
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from lalamo.module import LogicalAxis
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import ShardingConfig, is_sharded, reshard_as, sharding_of, with_sharding
from tests.helpers import make_sharding


def test_is_sharded_accepts_non_empty_named_sharding(fake_mesh: Mesh) -> None:
    sharding = make_sharding((LogicalAxis.BATCH,))

    assert is_sharded(sharding)
    assert sharding.mesh == fake_mesh


def test_is_sharded_rejects_none_and_empty_mesh_sharding() -> None:
    assert not is_sharded(None)
    assert not is_sharded(typeof(jnp.arange(4, dtype=jnp.float32)).sharding)


def test_sharding_config_constructors_use_full_physical_axis_names() -> None:
    devices = jax.devices("cpu")[:2]

    assert ShardingConfig.replicated(devices).resolve_axis(LogicalAxis.BATCH) is None
    assert ShardingConfig.data_parallel(devices).resolve_axis(LogicalAxis.BATCH) == "data"
    assert ShardingConfig.data_parallel(devices).resolve_axis(LogicalAxis.MATRIX) is None
    assert ShardingConfig.data_parallel(devices).resolve_axis(LogicalAxis.MIXTURE) is None
    assert ShardingConfig.data_parallel(devices).resolve_axis(LogicalAxis.SEQUENCE) is None
    assert ShardingConfig.expert_parallel(devices).resolve_axis(LogicalAxis.MIXTURE) == "expert"
    assert ShardingConfig.expert_parallel(devices).resolve_axis(LogicalAxis.BATCH) is None

    fsdp = ShardingConfig.fully_sharded_data_parallel(devices)
    assert fsdp.resolve_axis(LogicalAxis.BATCH) == "fsdp"
    assert fsdp.resolve_axis(LogicalAxis.MATRIX) == "fsdp"
    assert fsdp.resolve_axis(LogicalAxis.MIXTURE) == "fsdp"
    assert fsdp.resolve_axis(LogicalAxis.SEQUENCE) is None
    assert fsdp.resolve_axis(None) is None


def test_resolve_sharding_preserves_ranked_specs(fake_mesh: Mesh) -> None:
    del fake_mesh
    config = ShardingConfig.data_parallel(jax.devices("cpu")[:2])

    sharding = config.resolve_sharding((LogicalAxis.BATCH, None, LogicalAxis.SEQUENCE))

    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == config.mesh
    assert sharding.spec == PartitionSpec("data", None, None)


def test_resolve_sharding_returns_named_sharding_for_fully_replicated_requests() -> None:
    devices = jax.devices("cpu")[:2]

    replicated = ShardingConfig.replicated(devices).resolve_sharding((LogicalAxis.BATCH, LogicalAxis.MATRIX))
    data_parallel = ShardingConfig.data_parallel(devices).resolve_sharding((LogicalAxis.SEQUENCE, None))

    assert replicated == ShardingConfig.replicated(devices).make_sharding((None, None))
    assert data_parallel == ShardingConfig.data_parallel(devices).make_sharding((None, None))


def test_resolve_sharding_rejects_duplicate_physical_axes() -> None:
    config = ShardingConfig.fully_sharded_data_parallel(jax.devices("cpu")[:2])

    with pytest.raises(ValueError, match="same mesh axis"):
        config.resolve_sharding((LogicalAxis.BATCH, LogicalAxis.MATRIX))


def test_sharding_of_returns_concrete_array_sharding(fake_mesh: Mesh) -> None:
    array = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((LogicalAxis.BATCH,)))

    sharding = sharding_of(array)

    assert isinstance(sharding, NamedSharding)
    assert sharding == array.sharding
    assert sharding.mesh == fake_mesh


def test_with_sharding_updates_shape_dtype_struct_sharding(fake_mesh: Mesh) -> None:
    source = dummy_array((4,), jnp.float32, make_sharding((None,)))
    sharding = make_sharding((LogicalAxis.BATCH,))

    result = with_sharding(source, sharding)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == sharding
    assert result.sharding.mesh == fake_mesh


def test_reshard_as_uses_reference_named_sharding(fake_mesh: Mesh) -> None:
    source = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((LogicalAxis.MIXTURE,)))
    reference = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((LogicalAxis.BATCH,)))

    result = reshard_as(source, reference)

    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == reference.sharding
    assert result.sharding.mesh == fake_mesh
    np.testing.assert_array_equal(jax.device_get(result), jax.device_get(source))


def test_reshard_as_updates_replicated_named_sharding() -> None:
    source = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((LogicalAxis.MIXTURE,)))
    reference = jax.device_put(jnp.arange(4, dtype=jnp.float32), make_sharding((None,)))

    result = reshard_as(source, reference)

    assert result.sharding == reference.sharding
    assert jnp.array_equal(result, source)


def test_reshard_as_updates_shape_dtype_struct_to_reference_sharding(fake_mesh: Mesh) -> None:
    source = dummy_array((4,), jnp.float32, make_sharding((LogicalAxis.MIXTURE,)))
    reference = dummy_array((4,), jnp.float32, make_sharding((LogicalAxis.BATCH,)))

    result = reshard_as(source, reference)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == reference.sharding
    assert result.sharding.mesh == fake_mesh
    assert result.shape == source.shape
    assert result.dtype == source.dtype


def test_reshard_as_places_array_on_shape_dtype_struct_reference_sharding(fake_mesh: Mesh) -> None:
    source = jnp.arange(4, dtype=jnp.float32)
    reference = dummy_array((4,), jnp.float32, make_sharding((LogicalAxis.BATCH,)))

    result = reshard_as(source, reference)

    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding == reference.sharding
    assert result.sharding.mesh == fake_mesh
    np.testing.assert_array_equal(jax.device_get(result), jax.device_get(source))


def test_reshard_as_replicates_shape_dtype_struct_when_reference_has_none(fake_mesh: Mesh) -> None:
    source = dummy_array((4,), jnp.float32, make_sharding((LogicalAxis.MIXTURE,)))
    reference = dummy_array((4,), jnp.float32, make_sharding((None,)))

    result = reshard_as(source, reference)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding.mesh == fake_mesh
    assert result.sharding.spec == PartitionSpec(None)
    assert not is_sharded(result.sharding)
