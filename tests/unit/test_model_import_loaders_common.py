import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from lalamo.model_import.loaders.common import load_parameters
from lalamo.modules.common import field


class _LoaderModule(eqx.Module):
    weights: jax.Array = field()


def test_load_parameters_replaces_matching_arrays() -> None:
    module = _LoaderModule(weights=jnp.ones((2, 2), dtype=jnp.float32))

    loaded = load_parameters(
        lambda current: (current.weights,),
        module,
        [jnp.full((2, 2), 2.0, dtype=jnp.float32)],
    )

    assert loaded.weights.dtype == jnp.float32
    assert jnp.allclose(loaded.weights, jnp.full((2, 2), 2.0, dtype=jnp.float32))


def test_load_parameters_reports_leaf_name_on_dtype_mismatch() -> None:
    module = _LoaderModule(weights=jnp.ones((2, 2), dtype=jnp.bfloat16))

    with pytest.raises(ValueError, match=r"weights.*dtype bfloat16, got float32"):
        load_parameters(
            lambda current: (current.weights,),
            module,
            [jnp.full((2, 2), 2.0, dtype=jnp.float32)],
        )


def test_load_parameters_reports_leaf_name_on_shape_mismatch() -> None:
    module = _LoaderModule(weights=jnp.ones((2, 2), dtype=jnp.float32))

    with pytest.raises(ValueError, match=r"weights.*shape \(2, 2\).*\(3, 2\)"):
        load_parameters(
            lambda current: (current.weights,),
            module,
            [jnp.ones((3, 2), dtype=jnp.float32)],
        )
