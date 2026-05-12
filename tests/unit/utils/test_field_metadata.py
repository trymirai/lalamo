import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array

from lalamo.module import ParameterNorm, field
from lalamo.utils.field_metadata import field_metadata_for_leaf, partition_trainable


class _Child(eqx.Module):
    spectral: Array = field(norm=ParameterNorm.SPECTRAL)
    frozen: Array = field(trainable=False)


class _Root(eqx.Module):
    layers: tuple[_Child, ...]


def _root() -> _Root:
    return _Root((_Child(jnp.ones((2, 2)), jnp.ones((2,))),))


def test_field_metadata_follows_sequence_keys() -> None:
    metadata_by_name = {
        path[-1].name: field_metadata_for_leaf(_root(), path)
        for path, _leaf in jtu.tree_flatten_with_path(_root())[0]
    }

    assert metadata_by_name["spectral"].norm == ParameterNorm.SPECTRAL
    assert metadata_by_name["frozen"].trainable is False


def test_partition_trainable_follows_sequence_keys() -> None:
    trainable, frozen = partition_trainable(_root())

    assert trainable.layers[0].spectral is not None
    assert trainable.layers[0].frozen is None
    assert frozen.layers[0].spectral is None
    assert frozen.layers[0].frozen is not None
