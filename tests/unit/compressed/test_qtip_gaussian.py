from dataclasses import replace
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.incoherence import kronecker_transform
from lalamo.compressed.output_stack import OutputStackMatrix, OutputStackPart, OutputStackSpec
from lalamo.compressed.qtip_gaussian import (
    QtipGaussianMatrix,
    QtipGaussianSpec,
    TrellisCodebook,
)
from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.safetensors import safe_read, safe_write
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of
from lalamo.weight_matrix import Layout

pytestmark = pytest.mark.usefixtures("fake_mesh")

CODEBOOK = TrellisCodebook(1.0, (0.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("vector_width", [2, 4])
def test_saved_matrices_load_and_run_on_multiple_devices(tmp_path: Path, vector_width: Literal[2, 4]) -> None:
    config = ShardingConfig.fully_sharded_data_parallel(jax.devices("cpu"))
    with jax.set_mesh(config.mesh):
        spec = QtipGaussianSpec(vector_width, 8, 0, TrellisCodebook(0.125, (0.25, -0.5, 0.75, -1.0)))
        source = QtipGaussianMatrix(
            spec=spec,
            sharding_config=config,
            is_sharded=True,
            dtype_=jnp.float32,
            columns=128,
            codes=jnp.asarray(np.random.default_rng(0).integers(256, size=(8, spec.code_bytes(128)), dtype=np.uint8)),
            scales=jnp.linspace(0.5, 1, 8),
        )
        reference = np.asarray(source.decompress())
    exported = source.export()
    path = tmp_path / "weights.safetensors"
    with path.open("wb") as fd:
        safe_write(fd, exported.arrays)
    with jax.set_mesh(None), path.open("rb") as fd:
        _, arrays, _ = safe_read(fd)
        restored = (
            EmptyInitializer(None, config)
            .weight_matrix(8, 128)
            .load_exported(ExportResults(arrays, exported.metadata))
        )
        assert isinstance(restored, QtipGaussianMatrix)
        assert restored.spec == spec
        decoded = restored.decompress()
        np.testing.assert_array_equal(decoded, reference)
        assert sharding_of(decoded).spec[0] == config.resolve_axis(LogicalAxis.MATRIX)
        keychain = Keychain.init(0, sharding_config=config)
        vector = jax.device_put(np.linspace(-1, 1, 128, dtype=np.float32), config.make_sharding((None,)))
        np.testing.assert_allclose(restored.dot(vector, keychain=keychain), reference @ np.asarray(vector), atol=1e-4)
        tied = replace(restored, spec=replace(restored.spec, layout=Layout.INPUT_OUTPUT))
        indices = jax.device_put(
            np.arange(16, dtype=np.int32).reshape(8, 2) % 8,
            config.resolve_sharding((LogicalAxis.BATCH, None)),
        )
        rows = jax.jit(lambda index: tied.lookup_embedding(index, keychain=keychain))(indices)
        np.testing.assert_array_equal(rows, reference[np.asarray(indices)])
        assert sharding_of(rows).spec[0] == config.resolve_axis(LogicalAxis.BATCH)

        parts = (restored, replace(restored, scales=-restored.scales))
        stack_spec = OutputStackSpec(tuple(OutputStackPart(part.shape[0], part.spec) for part in parts))
        stack = OutputStackMatrix(spec=stack_spec, sharding_config=config, is_sharded=True, parts=parts)
        exported = stack.export()
        assert "part_rows" not in exported.arrays
        restored_stack = EmptyInitializer(None, config).weight_matrix(16, 128).load_exported(exported)
        np.testing.assert_array_equal(restored_stack.decompress(), np.concatenate((reference, -reference)))


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
        (QtipGaussianSpec(2, 4, 0, CODEBOOK), 34),
        (QtipGaussianSpec(2, 6, 0, CODEBOOK), 50),
        (QtipGaussianSpec(2, 8, 0, CODEBOOK), 65),
        (QtipGaussianSpec(4, 8, 0, CODEBOOK), 33),
        (QtipGaussianSpec(4, 6, 64, CODEBOOK), 32),
        (QtipGaussianSpec(4, 7, 64, CODEBOOK), 32),
        (QtipGaussianSpec(4, 8, 64, CODEBOOK), 48),
        (QtipGaussianSpec(4, 6, 128, CODEBOOK), 64),
        (QtipGaussianSpec(4, 7, 128, CODEBOOK), 64),
        (QtipGaussianSpec(4, 8, 128, CODEBOOK), 80),
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
        QtipGaussianSpec(*layout, codebook=CODEBOOK)
