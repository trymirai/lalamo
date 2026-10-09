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
    """Four rows fitted and packed by the independent Torch producer, and the weights it decoded them to."""
    layout = Layout.INPUT_OUTPUT if kind == LatticeKind.D4 else Layout.OUTPUT_INPUT
    spec = LatticeSpec(kind, layout)
    with np.load(DATA / "lattice_hyb036.npz") as data:
        if kind == LatticeKind.D4:
            table = jnp.asarray(data["table"])
        else:
            table = jnp.arange(-7, 8, 2, dtype=jnp.int8)[:, None]
        leaf = LatticeMatrix(
            spec=spec,
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=jnp.asarray(data[f"{kind}_codes"]),
            row_scales=jnp.asarray(data[f"{kind}_row_scale_bits"].view(jnp.bfloat16)),
            ladder_indices=jnp.asarray(data[f"{kind}_ladder_indices"]),
            ladder=jnp.asarray(data["ladder"]),
            table=table,
        ).astype(jnp.float32)
        signs = jnp.asarray(data["signs"])
        rotation = IncoherenceSigns(
            input_signs=None if kind == LatticeKind.D4 else signs,
            output_signs=signs if kind == LatticeKind.D4 else None,
        )
        return HybridMatrix.of(leaf, rotation, leaf.sharding_config), data[f"{kind}_expected"]


def test_lattice_matches_torch_fitted_rows() -> None:
    for kind in LatticeKind:
        matrix, expected = saved_rows(kind)
        if kind == LatticeKind.D4:
            expected = expected.T
        np.testing.assert_allclose(matrix.decompress(), expected, atol=2e-7, rtol=1e-6)


def test_d4_lookup_returns_the_torch_decoded_rows() -> None:
    matrix, expected = saved_rows(LatticeKind.D4)
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    for index in (2, jnp.array([3, 0, 1], dtype=jnp.int32)):
        actual = matrix.lookup_embedding(index, keychain=keychain)
        np.testing.assert_allclose(actual, expected[np.asarray(index)], atol=2e-7, rtol=1e-6)


def test_bf16_d4_rows_round_once_and_survive_to_full_precision() -> None:
    matrix, _ = saved_rows(LatticeKind.D4)
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    index = jnp.array([3, 0, 1], dtype=jnp.int32)
    rows = matrix.astype(jnp.bfloat16).lookup_embedding(index, keychain=keychain)
    np.testing.assert_array_equal(rows, matrix.lookup_embedding(index, keychain=keychain).astype(jnp.bfloat16))
    dense = matrix.astype(jnp.bfloat16).to_full_precision()
    np.testing.assert_array_equal(dense.lookup_embedding(index, keychain=keychain), rows)
