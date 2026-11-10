from dataclasses import replace
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import DTypeLike

from lalamo.compressed.hybrid import HybridMatrix
from lalamo.compressed.incoherence import IncoherenceProcessingMode, KroneckerProcessor, kronecker_transform
from lalamo.compressed.qtip_gaussian import (
    QtipGaussianMatrix,
    QtipGaussianSpec,
)
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.weight_matrix import Layout
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("scale_dtype", [jnp.float32, jnp.bfloat16])
def test_saved_tapes_decode_like_their_producers(scale_dtype: DTypeLike) -> None:
    # Four rows per tape, fitted and packed by the independent Torch producers for Qwen3.8 and Muse ("connected").
    config = make_test_sharding_config()
    with np.load(DATA / "qtip_gaussian_hyb036.npz") as hyb036, np.load(DATA / "qtip_gaussian_muse.npz") as muse:
        for data, name, spec, table in (
            (hyb036, "v2_k2", QtipGaussianSpec(2, 4, 0), hyb036["table_v2"]),
            (hyb036, "v2_k3", QtipGaussianSpec(2, 6, 0), hyb036["table_v2"]),
            (hyb036, "v4_k2", QtipGaussianSpec(4, 8, 64), hyb036["table_v4"]),
            (muse, "v4_k2_connected", QtipGaussianSpec(4, 8, 0), muse["v4_k2_connected_table"]),
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
                scales=jnp.asarray(data[f"{name}_scales"].astype(np.float32) * gains, dtype=scale_dtype),
                codebook=jnp.asarray(table),
            )
            rotated = data[f"{name}_rotated"]
            tolerance = max(1e-6, float(jnp.finfo(scale_dtype).eps))
            np.testing.assert_allclose(leaf.decompress(), rotated, rtol=tolerance, atol=1e-9, err_msg=name)

            rotation = KroneckerProcessor.of(
                jnp.asarray(data[f"{name}_signs"]).astype(jnp.int32), jnp.asarray(data[f"{name}_small_q"])
            )
            hybrid = HybridMatrix.of(leaf, rotation)
            dense = hybrid.decompress()
            vector = jax.device_put(jnp.linspace(-1, 1, leaf.shape[1]), config.make_sharding((None,)))
            actual = hybrid.dot(vector, keychain=Keychain.init(0, sharding_config=config))
            np.testing.assert_allclose(actual, dense @ vector, rtol=1e-5, atol=1e-5 * np.abs(dense).max())
            # bf16 trellis weights round once, after the rotation.
            np.testing.assert_array_equal(hybrid.astype(jnp.bfloat16).decompress(), dense.astype(jnp.bfloat16))
            # As a tied table (stored vocab x hidden): a lookup decodes and unrotates the hidden axis of its rows.
            tied = replace(leaf, spec=replace(spec, layout=Layout.INPUT_OUTPUT))
            hidden_rotation = KroneckerProcessor.of(
                rotation.signs, rotation.orthogonal_factor, IncoherenceProcessingMode.OUTPUT
            )
            hybrid = HybridMatrix.of(tied, hidden_rotation)
            rows, keychain = jnp.asarray([3, 0, 2]), Keychain.init(0, sharding_config=config)
            looked_up = hybrid.lookup_embedding(rows, keychain=keychain)
            np.testing.assert_array_equal(looked_up, hidden_rotation.unprocess_embedding(tied.decompress().T[rows]))
            template = EmptyInitializer(None, config).embedding_matrix(*leaf.shape)
            restored = template.load_exported(hybrid.export())
            assert isinstance(restored, HybridMatrix)
            assert isinstance(restored.quantized, QtipGaussianMatrix)
            assert restored.quantized.scales.dtype == scale_dtype
            np.testing.assert_array_equal(restored.decompress(), hybrid.decompress())


def test_kronecker_transform_matches_explicit_kronecker_product() -> None:
    values = np.arange(96, dtype=np.float32).reshape(4, 24) / 17
    q = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0].astype(np.float32)
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    rotation = np.kron(h / np.sqrt(np.float32(8)), q)
    np.testing.assert_allclose(kronecker_transform(jnp.asarray(values), jnp.asarray(q)), values @ rotation, atol=3e-6)


@pytest.mark.parametrize(
    ("spec", "row_bytes"),
    [
        (QtipGaussianSpec(2, 4, 0), 34),
        (QtipGaussianSpec(2, 6, 0), 50),
        (QtipGaussianSpec(2, 8, 0), 65),
        (QtipGaussianSpec(4, 8, 0), 33),
        (QtipGaussianSpec(4, 6, 64), 32),
        (QtipGaussianSpec(4, 7, 64), 32),
        (QtipGaussianSpec(4, 8, 64), 48),
        (QtipGaussianSpec(4, 6, 128), 64),
        (QtipGaussianSpec(4, 7, 128), 64),
        (QtipGaussianSpec(4, 8, 128), 80),
    ],
)
def test_tape_blocks_are_back_to_back_reversed_states(spec: QtipGaussianSpec, row_bytes: int) -> None:
    # Each block is its 16-bit start state, then each step's new low bits, reversed and packed LSB first; a restart
    # row pads to 16 bytes, a connected tape to a byte.
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
        tape += block[::-1]
    assert len(tape) <= 8 * row_bytes
    codes = jnp.asarray(np.packbits(np.fromiter(tape.ljust(8 * row_bytes, "0"), np.uint8), bitorder="little"))[None]
    assert spec.code_bytes(columns) == row_bytes
    np.testing.assert_array_equal(spec.states(codes, columns)[0], states)


@pytest.mark.parametrize("layout", [(2, 7, 0), (2, 4, 64), (4, 6, 0), (4, 4, 128)])
def test_unsupported_tape_layouts_fail(
    layout: tuple[Literal[2, 4], Literal[4, 6, 7, 8], Literal[0, 64, 128]],
) -> None:
    with pytest.raises(ValueError, match="Unsupported QTIP Gaussian layout"):
        QtipGaussianSpec(*layout)
