from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.hybrid import HybridMatrix, IncoherenceSigns
from lalamo.compressed.lattice import LatticeKind, LatticeMatrix, LatticeSpec
from lalamo.module import Keychain
from lalamo.weight_matrix import Layout
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

DATA = Path(__file__).parent / "data"


def saved_rows(kind: LatticeKind) -> tuple[HybridMatrix, np.ndarray]:
    # Four rows fitted and packed by the independent Torch producer, and the output-input weights it decoded.
    is_embedding = kind == LatticeKind.D4
    with np.load(DATA / "lattice_hyb036.npz") as data:
        leaf = LatticeMatrix(
            spec=LatticeSpec(kind, Layout.INPUT_OUTPUT if is_embedding else Layout.OUTPUT_INPUT),
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=jnp.asarray(data[f"{kind}_codes"]),
            row_scales=jnp.asarray(data[f"{kind}_row_scale_bits"].view(jnp.bfloat16)),
            ladder_indices=jnp.asarray(data[f"{kind}_ladder_indices"]),
            ladder=jnp.asarray(data["ladder"]),
            table=jnp.asarray(data["table"]) if is_embedding else jnp.arange(-7, 8, 2, dtype=jnp.int8)[:, None],
        ).astype(jnp.float32)
        signs = jnp.asarray(data["signs"])
        rotation = IncoherenceSigns(None, signs) if is_embedding else IncoherenceSigns(signs, None)
        expected = data[f"{kind}_expected"]
        return HybridMatrix.of(leaf, rotation, leaf.sharding_config), expected.T if is_embedding else expected


def test_lattice_matches_torch_fitted_rows() -> None:
    for kind in LatticeKind:
        matrix, expected = saved_rows(kind)
        np.testing.assert_allclose(matrix.decompress(), expected, atol=2e-7, rtol=1e-6)


def test_d4_lookup_rounds_once_and_survives_to_full_precision() -> None:
    matrix, expected = saved_rows(LatticeKind.D4)
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    for index in (2, jnp.array([3, 0, 1], dtype=jnp.int32)):
        rows = matrix.lookup_embedding(index, keychain=keychain)
        np.testing.assert_allclose(rows, expected.T[np.asarray(index)], atol=2e-7, rtol=1e-6)
    bf16_rows = matrix.astype(jnp.bfloat16).lookup_embedding(index, keychain=keychain)
    np.testing.assert_array_equal(bf16_rows, rows.astype(jnp.bfloat16))
    dense = matrix.astype(jnp.bfloat16).to_full_precision()
    np.testing.assert_array_equal(dense.lookup_embedding(index, keychain=keychain), bf16_rows)
