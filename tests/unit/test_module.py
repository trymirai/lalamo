from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding

from lalamo.compressed.int import IntMatrixForInference, IntMatrixForTraining, IntSpec
from lalamo.module import (
    Keychain,
    KeychainBroadcastMode,
    LalamoConfig,
    LalamoModule,
    ParameterNorm,
    ShardingAxis,
    field,
)
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import CompressionImplementation, FullPrecisionMatrix, WeightMatrix


def _key_data(*, key: jax.Array) -> tuple[int, ...]:
    return tuple(int(value) for value in jax.random.key_data(key).tolist())


def _flat_key_data(keys: jax.Array) -> list[tuple[int, ...]]:
    return [_key_data(key=key) for key in jnp.reshape(keys, (-1,))]


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


def test_keychain_broadcast_adds_distinct_trailing_keys() -> None:
    keychain = Keychain.init(0, shape=(2,))

    broadcast = keychain.broadcast((2, 3))

    assert broadcast.vmapped_keys.shape == (2, 3)
    assert jnp.array_equal(jax.random.key_data(broadcast.batch_key), jax.random.key_data(keychain.batch_key))
    assert len(set(_flat_key_data(broadcast.vmapped_keys))) == 6
    assert not jnp.array_equal(
        jax.random.key_data(broadcast.vmapped_keys[:, 0]),
        jax.random.key_data(broadcast.vmapped_keys[:, 1]),
    )


def test_keychain_broadcast_rejects_ambiguous_auto_mode() -> None:
    keychain = Keychain.init(0, shape=(2,))

    with pytest.raises(ValueError, match="Ambiguous"):
        keychain.broadcast((2, 2))

    assert keychain.broadcast((2, 2), mode=KeychainBroadcastMode.PREFIX).vmapped_keys.shape == (2, 2)
    assert keychain.broadcast((2, 2), mode=KeychainBroadcastMode.SUFFIX).vmapped_keys.shape == (2, 2)


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
    assert len({_key_data(key=split.batch_key) for split in splits}) == 3


def test_keychain_squeeze_removes_singleton_vmapped_axes() -> None:
    keychain = Keychain.init(0, shape=(1, 2, 1))

    squeezed = keychain.squeeze()

    assert squeezed.vmapped_keys.shape == (2,)
    assert jnp.array_equal(jax.random.key_data(squeezed.batch_key), jax.random.key_data(keychain.batch_key))
    assert jnp.array_equal(
        jax.random.key_data(squeezed.vmapped_keys),
        jax.random.key_data(keychain.vmapped_keys[0, :, 0]),
    )


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


class MatrixModule(LalamoModule[ExampleConfig]):
    matrix: WeightMatrix
    biases: jax.Array


class NestedModule(LalamoModule[ExampleConfig]):
    inner: MatrixModule


def _matrix_module() -> MatrixModule:
    weights = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 8
    matrix = IntSpec(bits=4, group_size=2).compress(weights, implementation=CompressionImplementation.TRAINING)
    return MatrixModule(
        config=ExampleConfig(width=4, dtype=jnp.dtype(jnp.float32)),
        matrix=matrix,
        biases=jnp.ones((4,), dtype=jnp.float32),
    )


def test_lalamo_module_astype_casts_weight_matrices_and_array_parameters(fake_mesh: Mesh) -> None:
    module = _matrix_module()

    result = module.astype(jnp.bfloat16)

    assert isinstance(module.matrix, IntMatrixForTraining)
    assert isinstance(module.matrix.scales.sharding, NamedSharding)
    assert module.matrix.scales.sharding.mesh == fake_mesh
    assert result.matrix.dtype == jnp.bfloat16
    assert result.biases.dtype == jnp.bfloat16
    assert module.matrix.dtype == jnp.float32
    assert module.biases.dtype == jnp.float32


def test_lalamo_module_astype_casts_dummy_array_parameters(fake_mesh: Mesh) -> None:
    module = ExampleModule(
        config=ExampleConfig(width=4, dtype=jnp.dtype(jnp.float32)),
        weights=dummy_array((4,), jnp.float32, make_sharding((ShardingAxis.DATA,))),
    )

    result = module.astype(jnp.bfloat16)

    assert result.weights.dtype == jnp.bfloat16
    assert isinstance(result.weights.sharding, NamedSharding)
    assert result.weights.sharding.mesh == fake_mesh
    assert result.weights.sharding == make_sharding((ShardingAxis.DATA,))


def test_lalamo_module_switch_implementation_recurses_into_nested_weight_matrices(fake_mesh: Mesh) -> None:
    module = NestedModule(
        config=ExampleConfig(width=4, dtype=jnp.dtype(jnp.float32)),
        inner=_matrix_module(),
    )

    result = module.switch_implementation(CompressionImplementation.INFERENCE)

    assert isinstance(module.inner.matrix, IntMatrixForTraining)
    assert isinstance(result.inner.matrix, IntMatrixForInference)
    assert isinstance(result.inner.matrix.scales.sharding, NamedSharding)
    assert result.inner.matrix.scales.sharding.mesh == fake_mesh


def test_lalamo_module_to_full_precision_converts_weight_matrices_without_casting(fake_mesh: Mesh) -> None:
    module = _matrix_module().astype(jnp.bfloat16)

    result = module.to_full_precision()

    assert isinstance(result.matrix, FullPrecisionMatrix)
    assert isinstance(result.matrix.weights.sharding, NamedSharding)
    assert result.matrix.weights.sharding.mesh == fake_mesh
    assert result.matrix.dtype == jnp.bfloat16
    assert result.biases.dtype == jnp.bfloat16
