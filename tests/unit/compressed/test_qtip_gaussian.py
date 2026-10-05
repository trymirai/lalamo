from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.qtip_gaussian import QtipGaussianMatrix, QtipGaussianSpec, full_rotation
from lalamo.compressed.utils.post_gains import GainAxis
from lalamo.module import Keychain
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

DATA = Path(__file__).parent / "data"

# Four rows per tape, fitted and packed by the independent Torch producers: Qwen3.8 tapes and Muse ("connected") tapes.
SAVED_TAPES = (
    ("v2_k2", QtipGaussianSpec(2, 4, 0)),
    ("v2_k3", QtipGaussianSpec(2, 6, 0)),
    ("v4_k2", QtipGaussianSpec(4, 8, 64)),
    ("v4_k2_connected", QtipGaussianSpec(4, 8, 0)),
    ("v2_k4_connected", QtipGaussianSpec(2, 8, 0)),
)


def saved_tape(name: str, spec: QtipGaussianSpec) -> tuple[QtipGaussianMatrix, np.ndarray]:
    is_muse = name.endswith("_connected")
    with np.load(DATA / ("qtip_gaussian_muse.npz" if is_muse else "qtip_gaussian_hyb036.npz")) as data:
        matrix = QtipGaussianMatrix(
            spec=spec,
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=spec.msb_first_codes(jnp.asarray(data[f"{name}_codes"]), data[f"{name}_signs"].shape[0]),
            scales=jnp.asarray(data[f"{name}_scales"]),
            gains=jnp.asarray(data[f"{name}_gains_bits"].view(jnp.bfloat16)),
            table=jnp.asarray(data[f"{name}_table"] if is_muse else data[f"table_v{spec.vector_width}"]),
            signs=jnp.asarray(data[f"{name}_signs"]),
            small_q=jnp.asarray(data[f"{name}_small_q"]),
        )
        return matrix, data[f"{name}_rotated"]


def test_saved_tapes_and_two_stage_scales_decode_exactly() -> None:
    for name, spec in SAVED_TAPES:
        matrix, expected = saved_tape(name, spec)
        np.testing.assert_array_equal(eqx.filter_jit(lambda m: m.rotated_weights())(matrix), expected)


def test_full_rotation_matches_explicit_kronecker_product() -> None:
    values = np.arange(96, dtype=np.float32).reshape(4, 24) / 17
    q = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0].astype(np.float32)
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    rotation = np.kron(h / np.sqrt(np.float32(8)), q)
    np.testing.assert_allclose(full_rotation(jnp.asarray(values), jnp.asarray(q)), values @ rotation, atol=3e-6)


def test_gain_stages_fold_before_the_rotation_and_round_after_it() -> None:
    original, rotated = saved_tape("v4_k2_connected", QtipGaussianSpec(4, 8, 0))
    with np.load(DATA / "post_gain_stages.npz") as data:
        row_gain = data["muse_up_gain"]
    pre_gain = np.array([0.995, 1.003, 1.004, 1.011], dtype=np.float32)
    column_gain = np.linspace(0.9, 1.1, original.shape[1], dtype=np.float32)
    matrix = replace(
        original,
        spec=QtipGaussianSpec(4, 8, 0, pre_gain_count=1, post_gain_axes=(GainAxis.ROW, GainAxis.COLUMN)),
        pre_gains=(jnp.asarray(pre_gain),),
        post_gains=(jnp.asarray(row_gain), jnp.asarray(column_gain)),
    )

    # Pre-gains multiply the saved rotated rows; each post-gain fold rounds to bfloat16 before the next one.
    expected_rotated = rotated * pre_gain[:, None]
    unrotated = np.asarray(full_rotation(jnp.asarray(expected_rotated), original.small_q)) * np.asarray(original.signs)
    expected = unrotated.astype(jnp.bfloat16)
    for gain in (row_gain[:, None], column_gain):
        expected = (expected.astype(np.float32) * gain).astype(jnp.bfloat16)
    np.testing.assert_array_equal(eqx.filter_jit(lambda m: m.rotated_weights())(matrix), expected_rotated)
    np.testing.assert_array_equal(matrix.decompress(), expected)
    vector = jnp.linspace(-1, 1, matrix.shape[1], dtype=jnp.float32)
    np.testing.assert_allclose(
        matrix.dot(vector, keychain=Keychain.init(0, sharding_config=matrix.sharding_config)),
        expected.astype(np.float32) @ np.asarray(vector),
        atol=2e-5,
        rtol=2e-5,
    )


def test_msb_first_codes_match_the_layout_uzu_reads() -> None:
    v4 = QtipGaussianSpec(4, 8, 64)
    package = jnp.asarray(np.frombuffer(bytes.fromhex("10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f 20"), np.uint8))
    stored = v4.msb_first_codes(package[None], 64)
    assert bytes(np.asarray(stored)[0]).hex(" ") == "11 10 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f 20"
    np.testing.assert_array_equal(v4.states(stored, 64)[0, :3], [0x1110, 0x1012, 0x1213])

    v2 = jnp.asarray(
        np.frombuffer(
            bytes.fromhex("da ca e3 44 bb 31 12 45 fd 6f 84 df 9a d7 c5 b3 d0 76 ac 0e 8f 53 a7 35 6c 88"), np.uint8
        )
    )
    np.testing.assert_array_equal(QtipGaussianSpec(2, 6, 0).states(v2[None], 64)[0, :2], [0xDACA, 0xB2B8])
