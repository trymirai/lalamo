import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from lalamo.module import ShardingAxis
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from lalamo.utils.sharding import make_sharding
from tests.common import assert_close_arrays, assert_named_sharding


def test_call_vmapped_accepts_added_sharding_axis(fake_mesh: Mesh) -> None:
    values = jax.device_put(jnp.arange(8, dtype=jnp.float32).reshape(2, 4), make_sharding((ShardingAxis.DATA, None)))

    result = call_vmapped(
        lambda value: value * 2,
        values,
        added_sharding_axis=ShardingAxis.DATA,
    )

    assert_close_arrays(result=result, reference=values * 2)
    assert_named_sharding(result.sharding, fake_mesh)
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

    assert_close_arrays(result=result, reference=values + 1)
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == values.sharding
