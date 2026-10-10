from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.hybrid import HybridMatrix, IncoherenceSigns
from lalamo.compressed.lattice import LatticeMatrix, LatticeSpec
from lalamo.module import Keychain
from lalamo.weight_matrix import Layout
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

DATA = Path(__file__).parent / "data"


def saved_rows() -> tuple[HybridMatrix, np.ndarray]:
    # Four rows fitted and packed by the independent Torch producer, and the output-input weights it decoded.
    with np.load(DATA / "lattice_hyb036.npz") as data:
        leaf = LatticeMatrix(
            spec=LatticeSpec(Layout.INPUT_OUTPUT),
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=jnp.asarray(data["d4_codes"]),
            row_scales=jnp.asarray(data["d4_row_scale_bits"].view(jnp.bfloat16)),
            ladder_indices=jnp.asarray(data["d4_ladder_indices"]),
            ladder=jnp.asarray(data["ladder"]),
            table=jnp.asarray(data["table"]),
        ).astype(jnp.float32)
        rotation = IncoherenceSigns(None, jnp.asarray(data["signs"]))
        return HybridMatrix.of(leaf, rotation, leaf.sharding_config), data["d4_expected"].T


def test_lattice_matches_torch_fitted_rows() -> None:
    matrix, expected = saved_rows()
    np.testing.assert_allclose(matrix.decompress(), expected, atol=2e-7, rtol=1e-6)


def test_d4_lookup_rounds_once_and_survives_to_full_precision() -> None:
    matrix, expected = saved_rows()
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    for index in (2, jnp.array([3, 0, 1], dtype=jnp.int32)):
        rows = matrix.lookup_embedding(index, keychain=keychain)
        np.testing.assert_allclose(rows, expected.T[np.asarray(index)], atol=2e-7, rtol=1e-6)
    bf16_rows = matrix.astype(jnp.bfloat16).lookup_embedding(index, keychain=keychain)
    np.testing.assert_array_equal(bf16_rows, rows.astype(jnp.bfloat16))
    dense = matrix.astype(jnp.bfloat16).to_full_precision()
    np.testing.assert_array_equal(dense.lookup_embedding(index, keychain=keychain), bf16_rows)
