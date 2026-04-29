import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, Sharding

from lalamo.module import ShardingAxis
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from lalamo.utils.sharding import make_sharding
from tests.common import assert_close


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def test_call_vmapped_accepts_added_sharding_axis(fake_mesh: Mesh) -> None:
    values = jax.device_put(jnp.arange(8, dtype=jnp.float32).reshape(2, 4), make_sharding((ShardingAxis.DATA, None)))

    result = call_vmapped(
        lambda value: value * 2,
        values,
        added_sharding_axis=ShardingAxis.DATA,
    )

    _assert_close(result=result, reference=values * 2)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == values.sharding


def test_call_vmapped_twice_accepts_added_sharding_axes(fake_mesh: Mesh) -> None:
    values = jax.device_put(
        jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
        make_sharding((ShardingAxis.DATA, None, None)),
    )

    result = call_vmapped_twice(
        lambda value: value + 1,
        values,
        added_sharding_axes=(ShardingAxis.DATA, None),
    )

    _assert_close(result=result, reference=values + 1)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == values.sharding
