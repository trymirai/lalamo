from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding

from lalamo.module import Keychain, LalamoConfig, LalamoModule, ParameterNorm, ShardingAxis, field
from lalamo.utils.sharding import make_sharding


def _key_data(key: jax.Array) -> tuple[int, ...]:
    return tuple(int(value) for value in jax.random.key_data(key).tolist())


def _flat_key_data(keys: jax.Array) -> list[tuple[int, ...]]:
    return [_key_data(key) for key in jnp.reshape(keys, (-1,))]


def test_keychain_init_is_deterministic_for_seed_and_shape() -> None:
    first = Keychain.init(0, shape=(2, 3))
    second = Keychain.init(0, shape=(2, 3))

    assert first.vmapped_keys.shape == (2, 3)
    assert first.batch_key.shape == ()
    assert jnp.array_equal(jax.random.key_data(first.vmapped_keys), jax.random.key_data(second.vmapped_keys))
    assert jnp.array_equal(jax.random.key_data(first.batch_key), jax.random.key_data(second.batch_key))


def test_keychain_init_produces_distinct_vmapped_keys() -> None:
    keychain = Keychain.init(0, shape=(2, 3))

    key_values = _flat_key_data(keychain.vmapped_keys)

    assert len(set(key_values)) == 6


def test_keychain_broadcast_adds_distinct_leading_keys() -> None:
    keychain = Keychain.init(0, shape=(2,))

    broadcast = keychain.broadcast((3, 2))

    assert broadcast.vmapped_keys.shape == (3, 2)
    assert jnp.array_equal(jax.random.key_data(broadcast.batch_key), jax.random.key_data(keychain.batch_key))
    assert len(set(_flat_key_data(broadcast.vmapped_keys))) == 6
    assert not jnp.array_equal(
        jax.random.key_data(broadcast.vmapped_keys[0]),
        jax.random.key_data(broadcast.vmapped_keys[1]),
    )


def test_keychain_broadcast_to_current_shape_preserves_keys() -> None:
    keychain = Keychain.init(0, shape=(2,))
    broadcast = keychain.broadcast((2,))

    assert jnp.array_equal(jax.random.key_data(broadcast.vmapped_keys), jax.random.key_data(keychain.vmapped_keys))
    assert jnp.array_equal(jax.random.key_data(broadcast.batch_key), jax.random.key_data(keychain.batch_key))


def test_keychain_broadcast_rejects_incompatible_shape() -> None:
    keychain = Keychain.init(0, shape=(2,))

    with pytest.raises(ValueError, match=r"Cannot broadcast|Expected target shape|Incompatible shapes"):
        keychain.broadcast((3,))


def test_keychain_broadcast_can_shard_vmapped_keys_without_changing_shape(fake_mesh: Mesh) -> None:
    keychain = Keychain.init(0, shape=(2,))

    sharded = keychain.broadcast((2,), sharding_axes=(ShardingAxis.DATA,))

    assert isinstance(sharded.vmapped_keys.sharding, NamedSharding)
    assert sharded.vmapped_keys.sharding.mesh == fake_mesh
    assert sharded.vmapped_keys.sharding == make_sharding((ShardingAxis.DATA,))
    assert jnp.array_equal(jax.random.key_data(sharded.batch_key), jax.random.key_data(keychain.batch_key))


def test_keychain_split_splits_vmapped_and_batch_keys() -> None:
    keychain = Keychain.init(0, shape=(2,))

    left, right = keychain.split()

    assert left.vmapped_keys.shape == (2,)
    assert right.vmapped_keys.shape == (2,)
    assert left.batch_key.shape == ()
    assert right.batch_key.shape == ()
    assert not jnp.array_equal(jax.random.key_data(left.vmapped_keys), jax.random.key_data(right.vmapped_keys))
    assert not jnp.array_equal(jax.random.key_data(left.batch_key), jax.random.key_data(right.batch_key))


def test_keychain_split_respects_requested_count() -> None:
    splits = Keychain.init(0, shape=(2,)).split(num=3)

    assert len(splits) == 3
    assert all(split.vmapped_keys.shape == (2,) for split in splits)
    assert len({_key_data(split.batch_key) for split in splits}) == 3


def test_keychain_rolling_broadcast_adds_distinct_leading_keys() -> None:
    keychain = Keychain.init(0, shape=(2,))

    broadcast = keychain.rolling_broadcast((3, 2))

    assert broadcast.vmapped_keys.shape == (3, 2)
    assert len(set(_flat_key_data(broadcast.vmapped_keys))) == 6
    assert not jnp.array_equal(
        jax.random.key_data(broadcast.vmapped_keys[0]),
        jax.random.key_data(broadcast.vmapped_keys[1]),
    )


@dataclass(frozen=True)
class ExampleConfig(LalamoConfig):
    width: int
    dtype: jnp.dtype


def test_lalamo_config_roundtrips_json() -> None:
    config = ExampleConfig(width=64, dtype=jnp.dtype(jnp.bfloat16))

    raw_config = config.to_json()

    assert raw_config == {
        "width": 64,
        "dtype": "bfloat16",
    }
    assert ExampleConfig.from_json(raw_config) == config


def test_field_rejects_static_trainable_field() -> None:
    with pytest.raises(ValueError, match="static and trainable"):
        field(static=True, trainable=True)


def test_field_metadata_records_trainability_and_norm() -> None:
    class Module(eqx.Module):
        trainable_weight: jax.Array = field()
        frozen_weight: jax.Array = field(trainable=False, norm=ParameterNorm.L_2)
        static_name: str = field(static=True)

    fields = Module.__dataclass_fields__

    assert fields["trainable_weight"].metadata["trainable"] is True
    assert fields["trainable_weight"].metadata["norm"] == ParameterNorm.L_INF
    assert fields["frozen_weight"].metadata["trainable"] is False
    assert fields["frozen_weight"].metadata["norm"] == ParameterNorm.L_2
    assert fields["static_name"].metadata["trainable"] is False


class ExampleModule(LalamoModule[ExampleConfig]):
    weights: jax.Array


def test_lalamo_module_config_is_static_and_weights_are_leaves() -> None:
    first = ExampleModule(config=ExampleConfig(width=1, dtype=jnp.dtype(jnp.float32)), weights=jnp.ones((2,)))
    second = ExampleModule(config=ExampleConfig(width=2, dtype=jnp.dtype(jnp.float32)), weights=jnp.ones((2,)))

    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)

    assert first_leaves == [first.weights]
    assert second_leaves == [second.weights]
    assert first_tree != second_tree
