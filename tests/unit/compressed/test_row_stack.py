from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.qtip_gaussian import QtipGaussianSpec
from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.utils.dummy_array import dummy_array
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


def test_qtip_row_stack_exports_one_rotation_and_round_trips() -> None:
    spec = QtipGaussianSpec(vector_width=2, transition_bits=4, restart_columns=0)
    sharding_config = make_test_sharding_config()
    template = spec.compress(
        dummy_array((2, 24), jnp.float32, sharding_config.resolve_sharding((None, None))),
        sharding_config=sharding_config,
    )
    signs = jnp.where(jnp.arange(24) % 2 == 0, 1.0, -1.0)
    small_q = jnp.eye(3, dtype=jnp.float32)
    parts = tuple(
        replace(
            template,
            codes=jnp.full((rows, template.codes.shape[1]), code, dtype=jnp.uint8),
            scales=jnp.ones((rows,), dtype=jnp.float32),
            codebook=jnp.arange(5, dtype=jnp.float32),
            signs=signs,
            small_q=small_q,
        )
        for rows, code in ((2, 0), (3, 1))
    )
    matrix = RowStackMatrix(
        spec=RowStackSpec(tuple((part.shape[0], part.spec) for part in parts)),
        sharding_config=sharding_config,
        is_sharded=True,
        parts=parts,
    )

    exported = matrix.export()
    assert {"signs", "small_q", "parts.0.codes", "parts.1.codes"} <= exported.arrays.keys()
    assert not {"parts.0.signs", "parts.0.small_q", "parts.1.signs", "parts.1.small_q"} & exported.arrays.keys()
    restored = matrix.load_exported(exported)
    np.testing.assert_array_equal(restored.decompress(), matrix.decompress())

    with pytest.raises(ValueError, match="must share signs and small_q"):
        replace(matrix, parts=(parts[0], replace(parts[1], signs=-signs))).export()
