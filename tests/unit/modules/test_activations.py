from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh
from jaxtyping import Array

from lalamo.module import ShardingAxis
from lalamo.modules.activations import GELU, Activation, Identity, SiLU
from lalamo.utils.sharding import make_sharding
from tests.common import assert_close_arrays, assert_named_sharding


def _sharded_input(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA,)))


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

    assert_close_arrays(result=result, reference=reference(inputs))
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding
