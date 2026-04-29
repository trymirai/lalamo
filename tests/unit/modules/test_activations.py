from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import ShardingAxis
from lalamo.modules.activations import GELU, Activation, Identity, SiLU
from lalamo.utils.sharding import make_sharding
from tests.common import assert_close


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_input(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA,)))


def _sharded_batched_inputs(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None, ShardingAxis.DATA)))


@pytest.mark.parametrize(
    ("activation", "reference"),
    [
        pytest.param(SiLU(alpha=0.75), lambda x: x / (1 + jnp.exp(-0.75 * x)), id="silu"),
        pytest.param(GELU(approximate=True), lambda x: jax.nn.gelu(x, approximate=True), id="approx-gelu"),
        pytest.param(GELU(approximate=False), lambda x: jax.nn.gelu(x, approximate=False), id="exact-gelu"),
        pytest.param(Identity(), lambda x: x, id="identity"),
    ],
)
def test_activation_matches_reference_under_jit_and_preserves_input_sharding(
    fake_mesh: Mesh,
    activation: Activation,
    reference: Callable[[Array], Array],
) -> None:
    inputs = _sharded_input(jnp.array([-2.0, -0.5, 0.5, 2.0], dtype=jnp.float32))

    result = eqx.filter_jit(lambda activation, values: activation(values))(activation, inputs)

    _assert_close(result=result, reference=reference(inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding


@pytest.mark.parametrize("activation", [SiLU(alpha=1.25), GELU(), Identity()])
def test_activation_vmapped_over_inputs_preserves_input_sharding(fake_mesh: Mesh, activation: Activation) -> None:
    inputs = _sharded_batched_inputs(
        jnp.array(
            [
                [-2.0, -0.5, 0.5, 2.0],
                [-1.0, -0.25, 0.25, 1.0],
            ],
            dtype=jnp.float32,
        ),
    )

    result = jax.vmap(activation)(inputs)
    reference = jax.vmap(activation)(jnp.asarray(jax.device_get(inputs)))

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding
