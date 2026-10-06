from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.qtip_gaussian import QtipGaussianMatrix, QtipGaussianSpec, full_rotation
from lalamo.model_import.loaders.packed_checkpoint import codebook_from_table
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


def production_table() -> np.ndarray:
    with np.load(DATA / "qtip_gaussian_muse.npz") as data:
        return data["v4_k2_connected_table"]


def saved_tape(name: str, spec: QtipGaussianSpec) -> QtipGaussianMatrix:
    """The saved tape with its two row scale stages folded into one, decoded through the production v4 table."""
    is_muse = name.endswith("_connected")
    with np.load(DATA / ("qtip_gaussian_muse.npz" if is_muse else "qtip_gaussian_hyb036.npz")) as data:
        scales = data[f"{name}_scales"].astype(np.float32)
        gains = data[f"{name}_gains_bits"].view(jnp.bfloat16).astype(np.float32)
        return QtipGaussianMatrix(
            spec=spec,
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            dtype_=jnp.bfloat16,
            codes=jnp.asarray(data[f"{name}_codes"]),
            scales=jnp.asarray(scales * gains),
            codebook=codebook_from_table(jnp.asarray(production_table()[:, : spec.vector_width])),
            signs=jnp.asarray(data[f"{name}_signs"]),
            small_q=jnp.asarray(data[f"{name}_small_q"]),
        )


def test_saved_tapes_decode_to_production_table_entries() -> None:
    # The hyb036 and v2_k4_connected fixture tables are not hash-affine, so every tape decodes through the v4 table.
    table = production_table()
    for name, spec in SAVED_TAPES:
        matrix = saved_tape(name, spec)
        states = np.asarray(spec.states(matrix.codes, matrix.shape[1]))
        expected = table[states, : spec.vector_width].reshape(matrix.shape) * np.asarray(matrix.scales)[:, None]
        decoded = eqx.filter_jit(lambda m: m.rotated_weights())(matrix)
        np.testing.assert_allclose(decoded, expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max(), err_msg=name)


def test_connected_v4_tape_matches_its_producer() -> None:
    matrix = saved_tape("v4_k2_connected", QtipGaussianSpec(4, 8, 0))
    with np.load(DATA / "qtip_gaussian_muse.npz") as data:
        producer_rotated = data["v4_k2_connected_rotated"]
    decoded = eqx.filter_jit(lambda m: m.rotated_weights())(matrix)
    np.testing.assert_allclose(decoded, producer_rotated, rtol=1e-6, atol=1e-6 * np.abs(producer_rotated).max())


def test_tables_that_are_not_computed_levels_are_rejected() -> None:
    with np.load(DATA / "qtip_gaussian_hyb036.npz") as data, pytest.raises(ValueError, match="scale \\* level"):
        codebook_from_table(jnp.asarray(data["table_v4"]))


def test_full_rotation_matches_explicit_kronecker_product() -> None:
    values = np.arange(96, dtype=np.float32).reshape(4, 24) / 17
    q = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0].astype(np.float32)
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    rotation = np.kron(h / np.sqrt(np.float32(8)), q)
    np.testing.assert_allclose(full_rotation(jnp.asarray(values), jnp.asarray(q)), values @ rotation, atol=3e-6)


def test_msb_first_states_match_the_layout_uzu_reads() -> None:
    v4 = QtipGaussianSpec(4, 8, 64)
    v4_codes = jnp.asarray(
        np.frombuffer(bytes.fromhex("11 10 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f 20"), np.uint8)
    )
    np.testing.assert_array_equal(v4.states(v4_codes[None], 64)[0, :3], [0x1110, 0x1012, 0x1213])

    v2_codes = jnp.asarray(
        np.frombuffer(
            bytes.fromhex("da ca e3 44 bb 31 12 45 fd 6f 84 df 9a d7 c5 b3 d0 76 ac 0e 8f 53 a7 35 6c 88"),
            np.uint8,
        )
    )
    np.testing.assert_array_equal(QtipGaussianSpec(2, 6, 0).states(v2_codes[None], 64)[0, :2], [0xDACA, 0xB2B8])
