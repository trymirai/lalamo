from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.compressed.s_surface import SSurfaceKind, SSurfaceMatrix, SSurfaceSpec
from lalamo.compressed.s_trellis import STrellisSpec
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import Layout
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.compressed.s_trellis_fixture import load_saved_trellis

pytestmark = pytest.mark.usefixtures("fake_mesh")


@pytest.mark.parametrize("surface_gate", [False, True])
def test_row_stack_preserves_parts_across_dot_transpose_and_reload(*, surface_gate: bool) -> None:
    config = make_test_sharding_config()
    parts = (
        load_saved_trellis("v2_k3", STrellisSpec(2, 6, 0)).astype(jnp.float32),
        load_saved_trellis("v4_k2", STrellisSpec(4, 8, 64)).astype(jnp.float32),
    )
    if surface_gate:
        with np.load(Path(__file__).parent / "data/s_surface_i4.npz") as data:
            gate = SSurfaceMatrix(
                spec=SSurfaceSpec(SSurfaceKind.I4, Layout.OUTPUT_INPUT),
                sharding_config=config,
                is_sharded=True,
                codes=jnp.asarray(data["codes"]),
                row_scales=jax.lax.bitcast_convert_type(jnp.asarray(data["row_scale_bits"]), jnp.bfloat16),
                ladder_indices=jnp.asarray(data["ladder_indices"]),
                ladder=jnp.asarray(data["ladder"]),
                table=jnp.arange(-15, 16, 2, dtype=jnp.int8)[:, None],
                signs=jnp.asarray(data["signs"]),
            )
        parts = (parts[0], gate.astype(jnp.float32).switch_sharding_config(config))
    stack = RowStackMatrix(
        spec=RowStackSpec(tuple((part.shape[0], part.spec) for part in parts)),
        parts=parts,
        sharding_config=config,
        is_sharded=True,
    )
    keychain = Keychain.init(0, sharding_config=config)
    x = jnp.linspace(-1, 1, stack.shape[1], dtype=jnp.float32)
    y = jnp.arange(stack.shape[0], dtype=jnp.float32)
    expected = jnp.concatenate(tuple(part.decompress() for part in parts))
    np.testing.assert_allclose(stack.dot(x, keychain=keychain), expected @ x, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(stack.dot(y, keychain=keychain, transposed=True), expected.T @ y, atol=2e-5, rtol=2e-5)
    template = stack.spec.compress(
        dummy_array(stack.shape, stack.dtype, make_sharding((None, None))), sharding_config=config
    )
    restored = template.load_exported(stack.export())
    for old, new in zip(jax.tree.leaves(stack), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(old, new)
