from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.hybrid import HybridMatrix, KroneckerRotation
from lalamo.compressed.qtip_gaussian import QtipGaussianMatrix, QtipGaussianSpec, codebook_from_table, full_rotation
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


def production_table() -> np.ndarray:
    with np.load(DATA / "qtip_gaussian_muse.npz") as data:
        return data["v4_k2_connected_table"]


def saved_tape(name: str, spec: QtipGaussianSpec) -> tuple[QtipGaussianMatrix, KroneckerRotation]:
    """The saved tape with its two row scale stages folded into one, decoded through the production v4 table."""
    is_muse = name.endswith("_connected")
    with np.load(DATA / ("qtip_gaussian_muse.npz" if is_muse else "qtip_gaussian_hyb036.npz")) as data:
        scales = data[f"{name}_scales"].astype(np.float32)
        gains = data[f"{name}_gains_bits"].view(jnp.bfloat16).astype(np.float32)
        signs = jnp.asarray(data[f"{name}_signs"])
        leaf = QtipGaussianMatrix(
            spec=spec,
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            dtype_=jnp.bfloat16,
            columns=signs.shape[0],
            codes=jnp.asarray(data[f"{name}_codes"]),
            scales=jnp.asarray(scales * gains),
            codebook=codebook_from_table(jnp.asarray(production_table()[:, : spec.vector_width])),
        )
        return leaf, KroneckerRotation(signs=signs, small_q=jnp.asarray(data[f"{name}_small_q"]))


def test_saved_tapes_decode_to_production_table_entries() -> None:
    # The hyb036 and v2_k4_connected fixture tables are not hash-affine, so every tape decodes through the v4 table.
    table = production_table()
    for name, spec in SAVED_TAPES:
        matrix, _ = saved_tape(name, spec)
        states = np.asarray(spec.states(matrix.codes, matrix.shape[1]))
        expected = table[states, : spec.vector_width].reshape(matrix.shape) * np.asarray(matrix.scales)[:, None]
        decoded = matrix.astype(jnp.float32).decompress()
        np.testing.assert_allclose(decoded, expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max(), err_msg=name)


def test_connected_v4_tape_matches_its_producer() -> None:
    leaf, rotation = saved_tape("v4_k2_connected", QtipGaussianSpec(4, 8, 0))
    with np.load(DATA / "qtip_gaussian_muse.npz") as data:
        producer_rotated = data["v4_k2_connected_rotated"]
    decoded = leaf.astype(jnp.float32).decompress()
    np.testing.assert_allclose(decoded, producer_rotated, rtol=1e-6, atol=1e-6 * np.abs(producer_rotated).max())

    # The hybrid applies kron(H, Q) and then the signs to the producer's rotated rows.
    expected = np.asarray(full_rotation(jnp.asarray(producer_rotated), rotation.small_q) * rotation.signs)
    tape = HybridMatrix.of(leaf.astype(jnp.float32), rotation, leaf.sharding_config)
    np.testing.assert_allclose(tape.decompress(), expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())

    vector = jax.device_put(jnp.linspace(-1, 1, expected.shape[1]), leaf.sharding_config.make_sharding((None,)))
    actual = tape.dot(vector, keychain=Keychain.init(0, sharding_config=leaf.sharding_config))
    np.testing.assert_allclose(actual, tape.decompress() @ vector, rtol=1e-5, atol=1e-5 * np.abs(expected).max())


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


@pytest.mark.parametrize(
    ("spec", "block_bytes"),
    [
        (QtipGaussianSpec(2, 4, 0), 34),
        (QtipGaussianSpec(2, 6, 0), 50),
        (QtipGaussianSpec(2, 8, 0), 65),
        (QtipGaussianSpec(4, 8, 0), 33),
        (QtipGaussianSpec(4, 6, 64), 14),
        (QtipGaussianSpec(4, 7, 64), 16),
        (QtipGaussianSpec(4, 8, 64), 17),
        (QtipGaussianSpec(4, 6, 128), 26),
        (QtipGaussianSpec(4, 7, 128), 30),
        (QtipGaussianSpec(4, 8, 128), 33),
    ],
)
def test_tape_blocks_are_byte_padded_msb_first_states(spec: QtipGaussianSpec, block_bytes: int) -> None:
    # Each block packs its 16-bit start state, then each step's new low bits, MSB first, padded to a byte.
    columns = 2 * spec.restart_columns or 128
    bits, block_columns = spec.transition_bits, spec.restart_columns or columns
    generator = np.random.default_rng(bits)
    states, tape = [], ""
    for _ in range(columns // block_columns):
        state = int(generator.integers(1 << 16))
        states.append(state)
        block = f"{state:016b}"
        for symbol in generator.integers(1 << bits, size=block_columns // spec.vector_width - 1):
            state = ((state << bits) | int(symbol)) & 0xFFFF
            states.append(state)
            block += f"{symbol:0{bits}b}"
        assert 0 <= 8 * block_bytes - len(block) < 8
        tape += block.ljust(8 * block_bytes, "0")
    codes = jnp.asarray(np.frombuffer(int(tape, 2).to_bytes(len(tape) // 8), np.uint8))[None]
    assert spec.code_bytes(columns) == len(tape) // 8
    np.testing.assert_array_equal(spec.states(codes, columns)[0], states)


def test_bf16_trellis_weights_round_once_after_the_rotation() -> None:
    leaf, rotation = saved_tape("v2_k3", QtipGaussianSpec(2, 6, 0))
    matrix = HybridMatrix.of(leaf, rotation, leaf.sharding_config)
    expected = matrix.astype(jnp.float32).decompress().astype(jnp.bfloat16)
    np.testing.assert_array_equal(matrix.astype(jnp.bfloat16).decompress(), expected)
