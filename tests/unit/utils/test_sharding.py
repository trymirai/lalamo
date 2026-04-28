import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding

from lalamo.module import ShardingAxis
from lalamo.utils.sharding import make_sharding, reshard_as


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
