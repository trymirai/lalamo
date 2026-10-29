from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jaxtyping import Array, DTypeLike

from lalamo.module import LogicalAxis
from lalamo.modules.activations import GELU, Activation, Identity, SiLU
from lalamo.utils.torch_interop import jax_to_torch
from tests.common import assert_close_arrays, assert_named_sharding
from tests.helpers import make_sharding


def _sharded_input(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH,)))


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


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float64])
def test_silu_matches_fused_torch_rounding_and_preserves_wide_precision(dtype: DTypeLike) -> None:
    with jax.enable_x64():
        inputs = jnp.linspace(-6, 6, 1000, dtype=jnp.float64).astype(dtype)
        reference = torch.nn.functional.silu(jax_to_torch(inputs)).to(torch.float64).numpy()
        result = jax.jit(SiLU())(inputs)
        assert result.dtype == inputs.dtype
        np.testing.assert_allclose(np.asarray(result).astype(np.float64), reference, atol=5e-15, rtol=5e-15)


@pytest.mark.parametrize("approximate", [False, True])
def test_gelu_bf16_matches_fused_torch_rounding(approximate: bool) -> None:
    inputs = jnp.linspace(-5, 5, 2048, dtype=jnp.bfloat16)
    reference = torch.nn.functional.gelu(jax_to_torch(inputs), approximate="tanh" if approximate else "none")
    result = jax.jit(GELU(approximate=approximate))(inputs)
    assert result.dtype == inputs.dtype
    np.testing.assert_allclose(np.asarray(result).astype(np.float32), reference.float().numpy(), atol=3e-6, rtol=1e-4)
