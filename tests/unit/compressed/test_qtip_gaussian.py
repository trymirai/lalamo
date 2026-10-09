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


def test_saved_tapes_decode_like_their_producers() -> None:
    # Four rows per tape, fitted and packed by the independent Torch producers for Qwen3.8 and Muse ("connected").
    # Each producer decoded through its own table; only the connected v4 table is the computed production codebook.
    config = make_test_sharding_config()
    with np.load(DATA / "qtip_gaussian_hyb036.npz") as hyb036, np.load(DATA / "qtip_gaussian_muse.npz") as muse:
        production = muse["v4_k2_connected_table"]
        with pytest.raises(ValueError, match="scale \\* level"):
            codebook_from_table(jnp.asarray(hyb036["table_v4"]))
        for data, name, spec, table in (
            (hyb036, "v2_k2", QtipGaussianSpec(2, 4, 0), hyb036["table_v2"]),
            (hyb036, "v2_k3", QtipGaussianSpec(2, 6, 0), hyb036["table_v2"]),
            (hyb036, "v4_k2", QtipGaussianSpec(4, 8, 64), hyb036["table_v4"]),
            (muse, "v4_k2_connected", QtipGaussianSpec(4, 8, 0), production),
            (muse, "v2_k4_connected", QtipGaussianSpec(2, 8, 0), muse["v2_k4_connected_table"]),
        ):
            gains = data[f"{name}_gains_bits"].view(jnp.bfloat16).astype(np.float32)
            leaf = QtipGaussianMatrix(
                spec=spec,
                sharding_config=config,
                is_sharded=True,
                dtype_=jnp.float32,
                columns=data[f"{name}_signs"].shape[0],
                codes=jnp.asarray(data[f"{name}_codes"]),
                scales=jnp.asarray(data[f"{name}_scales"].astype(np.float32) * gains),
                codebook=codebook_from_table(jnp.asarray(production[:, : spec.vector_width])),
            )
            states, scales = np.asarray(spec.states(leaf.codes, leaf.shape[1])), np.asarray(leaf.scales)[:, None]
            rotated = data[f"{name}_rotated"]
            np.testing.assert_allclose(table[states].reshape(leaf.shape) * scales, rotated, rtol=1e-6, atol=1e-9)
            expected = production[states, : spec.vector_width].reshape(leaf.shape) * scales
            np.testing.assert_allclose(leaf.decompress(), expected, rtol=1e-6, atol=1e-9, err_msg=name)

            rotation = KroneckerRotation(jnp.asarray(data[f"{name}_signs"]), jnp.asarray(data[f"{name}_small_q"]))
            hybrid = HybridMatrix.of(leaf, rotation, config)
            dense = hybrid.decompress()
            np.testing.assert_allclose(dense, full_rotation(leaf.decompress(), rotation.small_q) * rotation.signs)
            vector = jax.device_put(jnp.linspace(-1, 1, leaf.shape[1]), config.make_sharding((None,)))
            actual = hybrid.dot(vector, keychain=Keychain.init(0, sharding_config=config))
            np.testing.assert_allclose(actual, dense @ vector, rtol=1e-5, atol=1e-5 * np.abs(dense).max())
            # bf16 trellis weights round once, after the rotation.
            np.testing.assert_array_equal(hybrid.astype(jnp.bfloat16).decompress(), dense.astype(jnp.bfloat16))


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
