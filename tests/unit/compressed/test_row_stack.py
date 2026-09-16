import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import FullPrecisionSpec
from tests.helpers import make_sharding, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


def test_row_stack_preserves_parts_across_dot_transpose_and_reload() -> None:
    config = make_test_sharding_config()
    parts = tuple(
        FullPrecisionSpec().compress(jnp.arange(rows * 8, dtype=jnp.float32).reshape(rows, 8), sharding_config=config)
        for rows in (4, 2)
    )
    stack = RowStackMatrix(
        spec=RowStackSpec(tuple((part.shape[0], part.spec) for part in parts)),
        parts=parts,
        sharding_config=config,
        is_sharded=True,
    )
    keychain = Keychain.init(0, sharding_config=config)
    x, y = jnp.arange(8, dtype=jnp.float32), jnp.arange(6, dtype=jnp.float32)
    expected = jnp.concatenate(tuple(part.decompress() for part in parts))
    np.testing.assert_array_equal(stack.dot(x, keychain=keychain), expected @ x)
    np.testing.assert_array_equal(stack.dot(y, keychain=keychain, transposed=True), expected.T @ y)
    template = stack.spec.compress(
        dummy_array(stack.shape, stack.dtype, make_sharding((None, None))), sharding_config=config
    )
    restored = template.load_exported(stack.export())
    for old, new in zip(jax.tree.leaves(stack), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(old, new)
