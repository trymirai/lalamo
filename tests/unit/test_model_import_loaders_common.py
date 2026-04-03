from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from lalamo.model_import.loaders.common import load_parameters
from lalamo.modules.common import field, shard_batch_axis


class _LoaderModule(eqx.Module):
    weights: jax.Array = field()


class _FakeShardingConfig:
    data_axis_size = 4
    data_axis_name = "data"

    def make_sharding(self, pspec: object) -> object:
        return pspec


def test_load_parameters_replaces_matching_arrays() -> None:
    module = _LoaderModule(weights=jnp.ones((2, 2), dtype=jnp.float32))

    loaded = load_parameters(
        lambda current: (current.weights,),
        module,
        (jnp.full((2, 2), 2.0, dtype=jnp.float32),),
    )

    assert loaded.weights.dtype == jnp.float32
    assert jnp.allclose(loaded.weights, jnp.full((2, 2), 2.0, dtype=jnp.float32))


@pytest.mark.parametrize(
    ("module", "new_value"),
    [
        (
            _LoaderModule(weights=jnp.ones((2, 2), dtype=jnp.bfloat16)),
            jnp.full((2, 2), 2.0, dtype=jnp.float32),
        ),
        (
            _LoaderModule(weights=jnp.ones((2, 2), dtype=jnp.float32)),
            jnp.ones((3, 2), dtype=jnp.float32),
        ),
    ],
)
def test_load_parameters_rejects_incompatible_replacements(
    module: _LoaderModule,
    new_value: jax.Array,
) -> None:
    with pytest.raises(ValueError, match="preserve the selected parameter structure"):
        load_parameters(
            lambda current: (current.weights,),
            module,
            (new_value,),
        )


def test_shard_batch_axis_pads_higher_rank_prng_key_arrays() -> None:
    keys = jax.random.split(jax.random.key(0), 4).reshape(2, 2)

    with patch("lalamo.modules.common.jax.lax.with_sharding_constraint", side_effect=lambda array, _sharding: array):
        sharded = shard_batch_axis(keys, _FakeShardingConfig(), batch_axis=1)

    assert sharded.shape == (2, 4)
