import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from lalamo.module import Keychain, KeychainBroadcastMode, LogicalAxis
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from tests.common import assert_close_arrays, assert_named_sharding
from tests.helpers import make_sharding, make_test_sharding_config


def test_call_vmapped_accepts_added_sharding_axis(fake_mesh: Mesh) -> None:
    values = jax.device_put(jnp.arange(8, dtype=jnp.float32).reshape(2, 4), make_sharding((LogicalAxis.BATCH, None)))

    result = call_vmapped(
        lambda value: value * 2,
        values,
        added_sharding_axis=make_test_sharding_config().resolve_axis(LogicalAxis.BATCH),
    )

    assert_close_arrays(result=result, reference=values * 2)
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == values.sharding


def test_call_vmapped_twice_accepts_added_sharding_axes(fake_mesh: Mesh) -> None:
    values = jax.device_put(
        jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
        make_sharding((LogicalAxis.BATCH, None, None)),
    )

    result = call_vmapped_twice(
        lambda value: value + 1,
        values,
        added_sharding_axes=(make_test_sharding_config().resolve_axis(LogicalAxis.BATCH), None),
    )

    assert_close_arrays(result=result, reference=values + 1)
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == values.sharding


def test_call_vmapped_twice_broadcasts_outer_keychain_over_inner_axis() -> None:
    values = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)
    keychain = Keychain.init(0, shape=(1,), sharding_config=make_test_sharding_config())

    result = call_vmapped_twice(
        lambda _value, *, keychain: jax.random.key_data(keychain.vmapped_keys),
        values,
        keychain=keychain,
    )
    expected_keychain = keychain.broadcast((1, 3), mode=KeychainBroadcastMode.SUFFIX)

    assert jnp.array_equal(result, jax.random.key_data(expected_keychain.vmapped_keys))
