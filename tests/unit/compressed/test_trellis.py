import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.trellis import (
    TapeLayout,
    TrellisSpec,
    codebook_scale,
    level_table,
    search_scales,
    states_to_levels,
    states_to_tape,
    tape_to_states,
    window_codebook,
)
from lalamo.preconditioner import Preconditioner
from tests.common import assert_close_arrays
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

# Dumped from the 2**16-state table uzu materializes from its trellis_format.rs (PR 808).
MATERIALIZED_LEVEL_SUM = 401611
MATERIALIZED_LEVELS = {
    0: (27, 5, -13, -17),
    1: (55, -13, -27, -3),
    2: (-27, 0, 4, -8),
    255: (20, -3, 2, 19),
    256: (39, 3, 31, 25),
    4097: (1, 15, -28, -14),
    12345: (-9, -30, 9, 27),
    32768: (-13, 2, -46, -46),
    65535: (8, 19, 21, -27),
}

# One L = 32, k = 3 row from uzu's trellis_format_test.rs: the packed bytes and the states read out of them.
GOLDEN_TAPE = "ddaff0c82b87c44e1b5f5b3e1155aa14f1fc7060c927348522e27966"
GOLDEN_STATES = (
    0x6679E222,
    0x9E222853,
    0x22853427,
    0x53427C96,
    0x27C96070,
    0x96070FCF,
    0x70FCF114,
    0xCF114AA5,
    0x14AA5511,
    0xA55113E5,
    0x113E5B5F,
    0xE5B5F1B4,
    0x5F1B4EC4,
    0xB4EC4872,
    0xC4872BC8,
    0x72BC8F0A,
    0xC8F0AFDD,
)


def ramp_weights(rows: int = 16, cols: int = 16) -> jax.Array:
    return (jnp.arange(rows * cols, dtype=jnp.float32).reshape(rows, cols) - 131) / 37


def manual_decode(spec: TrellisSpec, packed_tape: np.ndarray, scales: np.ndarray, cols: int) -> np.ndarray:
    steps_per_block = spec.restart_columns // 4
    states = np.zeros((packed_tape.shape[0], cols // 4), dtype=np.uint32)
    for row_index, row in enumerate(packed_tape):
        bits = np.unpackbits(row, bitorder="little")
        for block in range(cols // spec.restart_columns):
            for step in range(steps_per_block):
                offset = block * spec.block_bits + (steps_per_block - 1 - step) * spec.code_bits
                window = bits[offset : offset + spec.window_bits]
                states[row_index, block * steps_per_block + step] = sum(
                    int(bit) << index for index, bit in enumerate(window)
                )
    levels = np.asarray(states_to_levels(jnp.asarray(states)), dtype=np.float64).reshape(packed_tape.shape[0], cols)
    return levels * codebook_scale() * scales[:, None]


def test_trellis_state_levels_match_uzu_materialized_table() -> None:
    levels = states_to_levels(jnp.arange(1 << 16, dtype=jnp.uint32))

    assert int(jnp.sum(levels.astype(jnp.int32))) == MATERIALIZED_LEVEL_SUM
    for state, expected in MATERIALIZED_LEVELS.items():
        assert tuple(int(level) for level in levels[state]) == expected


def test_trellis_level_table_is_unit_variance_after_scaling() -> None:
    levels = np.array(level_table(), dtype=np.float64)

    assert np.mean(np.square(levels * codebook_scale())) == pytest.approx(1.0, abs=1e-6)


def test_trellis_tape_layout_matches_uzu_golden_tape() -> None:
    layout = TapeLayout(TrellisSpec(bits=3, window_bits=32, restart_columns=68), cols=68)
    golden_tape = jnp.asarray(np.frombuffer(bytes.fromhex(GOLDEN_TAPE), dtype=np.uint8))[None]
    golden_states = jnp.asarray(np.array(GOLDEN_STATES, dtype=np.uint32))[None]

    assert jnp.array_equal(tape_to_states(golden_tape, layout), golden_states)
    assert jnp.array_equal(states_to_tape(golden_states, layout), golden_tape)


def test_trellis_viterbi_matches_brute_force_search() -> None:
    spec = TrellisSpec(bits=1, window_bits=8, restart_columns=8)
    weights = ramp_weights()
    codebook = np.asarray(window_codebook(spec.window_bits), dtype=np.float64)
    headers = np.arange(1 << spec.window_bits)
    successors = ((headers[:, None] << 4) & 0xFF) | np.arange(16)[None, :]
    path_codewords = np.concatenate(
        [np.broadcast_to(codebook[headers][:, None, :], (256, 16, 4)), codebook[successors]],
        axis=-1,
    )

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    targets = np.asarray(weights / search_scales(weights)[:, None], dtype=np.float64)
    rows, cols = weights.shape
    layout = TapeLayout(spec, cols)
    fitted_codewords = codebook[np.asarray(tape_to_states(matrix.packed_tape, layout))].reshape(rows, cols)
    fitted_costs = np.sum(np.square(targets - fitted_codewords), axis=-1)
    for row in range(rows):
        block_costs = [
            np.min(np.sum(np.square(targets[row, block * 8 : (block + 1) * 8] - path_codewords), axis=-1))
            for block in range(layout.blocks)
        ]
        assert fitted_costs[row] == pytest.approx(sum(block_costs), rel=1e-5)


def test_trellis_decompress_matches_manual_decode() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)

    matrix = spec.compress(ramp_weights(8), sharding_config=make_test_sharding_config())

    reference = manual_decode(spec, np.asarray(matrix.packed_tape), np.asarray(matrix.scales), matrix.cols)
    assert_close_arrays(result=matrix.decompress(), reference=jnp.asarray(reference, dtype=jnp.float32))


def test_trellis_bfloat16_decode_rounds_once_after_the_float32_scale_product() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    matrix = spec.compress(ramp_weights().astype(jnp.bfloat16), sharding_config=make_test_sharding_config())

    decompressed = matrix.decompress()

    states = tape_to_states(matrix.packed_tape, TapeLayout(spec, matrix.cols))
    levels = np.asarray(states_to_levels(states), dtype=np.float32).reshape(matrix.shape)
    folded_scales = np.asarray(matrix.scales, dtype=np.float32) * np.float32(codebook_scale())
    assert decompressed.dtype == jnp.bfloat16
    assert np.array_equal(np.asarray(decompressed), (levels * folded_scales[:, None]).astype(jnp.bfloat16))


def test_trellis_input_preconditioner_lowers_the_input_weighted_error() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    weights = jax.random.normal(jax.random.key(3), (16, 32), dtype=jnp.float32)
    factor = jax.random.normal(jax.random.key(7), (32, 32), dtype=jnp.float32)
    input_block = factor @ factor.T + jnp.identity(32, dtype=jnp.float32) * 32

    plain = spec.compress(weights, sharding_config=make_test_sharding_config())
    preconditioned = spec.compress(
        weights,
        preconditioner=Preconditioner.init(input_block=input_block),
        sharding_config=make_test_sharding_config(),
    )

    def input_weighted_error(decompressed: jax.Array) -> float:
        residual = np.asarray(weights, dtype=np.float64) - np.asarray(decompressed, dtype=np.float64)
        return float(np.einsum("oi,ij,oj->", residual, np.asarray(input_block, dtype=np.float64), residual))

    assert input_weighted_error(preconditioned.decompress()) < input_weighted_error(plain.decompress())
