from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed import U4M3Z4Matrix, U4M3Z4Spec
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import ShapeDtypeSpec, WeightMatrixSpec
from tests.common import assert_close_arrays
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _matrix(group_size: Literal[32, 64], leading_dims: tuple[int, ...] = ()) -> U4M3Z4Matrix:
    packed_weights = jnp.full((*leading_dims, 2, group_size), 0xF0, dtype=jnp.uint8)
    packed_weights = packed_weights.at[..., group_size // 2].set(0xF1)
    base_scales = (jnp.arange(prod((*leading_dims, 2)), dtype=jnp.float32) + 1).reshape((*leading_dims, 2)) / 10
    mz_plane = jnp.zeros((*leading_dims, 2, 2), dtype=jnp.uint8)
    mz_plane = mz_plane.at[..., 0, 0].set(0x7F).at[..., 0, 1].set(0x20)
    return U4M3Z4Spec(group_size=group_size).from_packed_parameters(
        packed_weights=packed_weights,
        base_scales=base_scales,
        mz_plane=mz_plane,
        sharding_config=make_test_sharding_config(),
    )


@pytest.mark.parametrize(("group_size", "leading_dims"), [(32, ()), (64, ()), (32, (2,)), (64, (2,))])
def test_decompress_dot_export_and_shape_template_load(
    group_size: Literal[32, 64],
    leading_dims: tuple[int, ...],
) -> None:
    matrix = _matrix(group_size, leading_dims)
    values = jax.device_get(matrix.decompress())
    scales = jax.device_get(matrix.base_scales)
    assert_close_arrays(result=values[..., 0, 0], reference=-120 * scales[..., 0])
    assert_close_arrays(result=values[..., 0, group_size], reference=3 * scales[..., 0])
    if not leading_dims:
        vector = jnp.arange(matrix.shape[-1], dtype=jnp.float32)
        keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
        assert_close_arrays(result=matrix.dot(vector, keychain=keychain), reference=values @ vector)
        assert_close_arrays(
            result=matrix.dot(jnp.array([2.0, -1.0]), keychain=keychain, transposed=True),
            reference=values.T @ jnp.array([2.0, -1.0]),
        )

    exported = matrix.export()
    assert set(exported.arrays) == {"weights", "scales", "mz_plane"}
    assert (jax.device_get(exported.arrays["weights"])[..., 0, 0] == 0xF0).all()
    assert (jax.device_get(exported.arrays["mz_plane"])[..., 0, 0] == 0x7F).all()
    assert WeightMatrixSpec.from_json(exported.metadata["spec"]) == matrix.spec
    config = matrix.sharding_config
    dummy = dummy_array(matrix.shape, jnp.float32, config.resolve_sharding((None,) * len(matrix.shape)))
    template = ShapeDtypeSpec().compress(dummy, sharding_config=config)
    restored = template.load_exported(exported)
    assert_close_arrays(result=restored.decompress(), reference=matrix.decompress())

    half = matrix.astype(jnp.float16)
    assert half.base_scales.dtype == jnp.float32
    assert half.decompress().dtype == jnp.float16
    assert_close_arrays(result=half.decompress(), reference=matrix.decompress().astype(jnp.float16))
    half_value = jax.device_get(half.decompress())[..., 0, group_size]
    assert (half_value == (scales[..., 0] * 3).astype(jnp.float16)).all()
    assert (half_value != (scales[..., 0].astype(jnp.float16) * 3).astype(jnp.float16)).all()
