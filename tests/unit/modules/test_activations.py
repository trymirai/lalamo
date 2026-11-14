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
from tests.common import assert_close_arrays, assert_named_sharding
from tests.helpers import make_sharding


@pytest.mark.parametrize("approximate", [True, False])
@pytest.mark.parametrize(
    ("jax_dtype", "torch_dtype"),
    [(jnp.float16, torch.float16), (jnp.bfloat16, torch.bfloat16)],
)
def test_low_precision_gelu_matches_torch(
    approximate: bool,
    jax_dtype: DTypeLike,
    torch_dtype: torch.dtype,
) -> None:
    inputs = jnp.linspace(-8, 8, 4096, dtype=jnp.float32).astype(jax_dtype)
    reference = torch.nn.functional.gelu(
        torch.from_numpy(np.asarray(inputs).astype(np.float32)).to(torch_dtype),
        approximate="tanh" if approximate else "none",
    )
    result = np.asarray(eqx.filter_jit(GELU(approximate=approximate))(inputs)).astype(np.float32)
    reference = reference.float().numpy()
    relative_error = np.linalg.norm(result - reference) / np.linalg.norm(reference)
    assert relative_error < 1e-5
    # Float32 library rounding can land on opposite sides of a low-precision midpoint.
    np.testing.assert_allclose(
        result,
        reference,
        atol=1e-6,
        rtol=float(jnp.finfo(jax_dtype).eps),
    )


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
