from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.lattice import LatticeKind, LatticeMatrix, LatticeSpec
from lalamo.compressed.utils.post_gains import GainAxis
from lalamo.module import Keychain
from lalamo.weight_matrix import Layout
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

DATA = Path(__file__).parent / "data"


def saved_rows(kind: LatticeKind) -> tuple[LatticeMatrix, np.ndarray]:
    """Four rows fitted and packed by the independent Torch producer, and the weights it decoded them to."""
    layout = Layout.INPUT_OUTPUT if kind == LatticeKind.D4 else Layout.OUTPUT_INPUT
    prefix = "" if kind == LatticeKind.I4 else f"{kind}_"
    spec = LatticeSpec(kind, layout)
    states = 1 << spec.code_bits
    with np.load(DATA / ("lattice_i4.npz" if kind == LatticeKind.I4 else "lattice_hyb036.npz")) as data:
        if kind == LatticeKind.D4:
            table = jnp.asarray(data["table"])
        else:
            table = jnp.arange(1 - states, states, 2, dtype=jnp.int8)[:, None]
        matrix = LatticeMatrix(
            spec=spec,
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=jnp.asarray(data[f"{prefix}codes"]),
            row_scales=jax.lax.bitcast_convert_type(jnp.asarray(data[f"{prefix}row_scale_bits"]), jnp.bfloat16),
            ladder_indices=jnp.asarray(data[f"{prefix}ladder_indices"]),
            ladder=jnp.asarray(data["ladder"]),
            table=table,
            signs=jnp.asarray(data["signs"]),
        ).astype(jnp.float32)
        return matrix, data[f"{prefix}expected"]


def test_lattice_matches_torch_fitted_rows() -> None:
    for kind in LatticeKind:
        matrix, expected = saved_rows(kind)
        if matrix.spec.layout == Layout.INPUT_OUTPUT:
            expected = expected.T
        np.testing.assert_allclose(matrix.decompress(), expected, atol=2e-7, rtol=1e-6)


def test_d4_lookup_returns_the_torch_decoded_rows() -> None:
    matrix, expected = saved_rows(LatticeKind.D4)
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    for index in (2, jnp.array([3, 0, 1], dtype=jnp.int32)):
        actual = matrix.lookup_embedding(index, keychain=keychain)
        np.testing.assert_allclose(actual, expected[np.asarray(index)], atol=2e-7, rtol=1e-6)


def test_i4_gain_stages_match_torch_bf16_weights() -> None:
    matrix, _ = saved_rows(LatticeKind.I4)
    with np.load(DATA / "post_gain_stages.npz") as data:
        matrix = replace(
            matrix,
            spec=replace(matrix.spec, post_gain_axes=(GainAxis.ROW, GainAxis.ROW, GainAxis.ROW, GainAxis.COLUMN)),
            post_gains=tuple(jnp.asarray(data[name]) for name in ("i4_t0", "i4_t5", "i4_row_gain", "i4_column_gain")),
        )
        expected = jax.lax.bitcast_convert_type(jnp.asarray(data["i4_expected_bits"]), jnp.bfloat16)
    # The producer's dense H32 matmul leaves tiny residuals at exact cancellation zeros.
    actual = np.asarray(matrix.decompress())
    nonzero = actual != 0
    np.testing.assert_array_equal(actual[nonzero], np.asarray(expected)[nonzero])
    np.testing.assert_allclose(actual[~nonzero], np.asarray(expected)[~nonzero], atol=1e-8, rtol=0)
    inputs = jnp.linspace(-1, 1, matrix.shape[1], dtype=jnp.float32)
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    np.testing.assert_allclose(matrix.dot(inputs, keychain=keychain), expected.astype(jnp.float32) @ inputs, atol=2e-5)
