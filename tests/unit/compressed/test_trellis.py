from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, Sharding

from lalamo.compressed.hybrid import HybridMatrix, HybridSpec
from lalamo.compressed.trellis import (
    TrellisMatrix,
    TrellisSpec,
    _codebook,
    _codebook_scale,
    _level_table,
    _search_scales,
    _states_to_codewords,
    _states_to_levels,
    _states_to_tape,
    _tape_to_states,
    _TapeLayout,
    _weights_to_states,
)
from lalamo.module import Keychain, LogicalAxis
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import ShardingConfig, is_sharded
from lalamo.weight_matrix import Layout, WeightMatrixSpec
from tests.common import assert_close_arrays, assert_named_sharding
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.compressed import test_common as compressed_common

pytestmark = pytest.mark.usefixtures("fake_mesh")

# Dumped from the 2**16-state table uzu materializes from its trellis_format.rs (PR 808).
_MATERIALIZED_LEVEL_SUM = 401611
_MATERIALIZED_LEVELS = {
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
_GOLDEN_TAPE = "ddaff0c82b87c44e1b5f5b3e1155aa14f1fc7060c927348522e27966"
_GOLDEN_STATES = (
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


def _logical_weights(*leading_dims: int) -> jax.Array:
    shape = (*leading_dims, 16, 16)
    return (jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape) - 131) / 37


def _gaussian_weights(rows: int, cols: int) -> jax.Array:
    return jax.random.normal(jax.random.key(3), (rows, cols), dtype=jnp.float32)


def _put_on_sharding(matrix: TrellisMatrix, tape_sharding: Sharding, scale_sharding: Sharding) -> TrellisMatrix:
    return eqx.tree_at(
        lambda m: (m.packed_tape, m.scales),
        matrix,
        (jax.device_put(matrix.packed_tape, tape_sharding), jax.device_put(matrix.scales, scale_sharding)),
    )


def _recurrence_states(spec: TrellisSpec, headers: np.ndarray, codes: np.ndarray) -> np.ndarray:
    code_bits = 4 * spec.bits
    state_mask = (1 << spec.window_bits) - 1
    *_, steps_per_block = codes.shape
    states = np.zeros((*headers.shape, steps_per_block + 1), dtype=np.uint64)
    states[..., 0] = headers
    for step in range(steps_per_block):
        states[..., step + 1] = ((states[..., step] << code_bits) & state_mask) | codes[..., step]
    return states.reshape(*headers.shape[:-1], -1).astype(np.uint32)


def _manual_bit_field(row: np.ndarray, bit_offset: int, width: int) -> int:
    value = 0
    for bit in range(width):
        position = bit_offset + bit
        value |= ((int(row[position // 8]) >> (position % 8)) & 1) << bit
    return value


def _manual_decode(spec: TrellisSpec, packed_tape: np.ndarray, scales: np.ndarray, cols: int) -> np.ndarray:
    steps_per_block = spec.restart_columns // 4
    block_bits = spec.window_bits + (steps_per_block - 1) * spec.code_bits
    states = np.zeros((packed_tape.shape[0], cols // 4), dtype=np.uint32)
    for row_index, row in enumerate(packed_tape):
        for block in range(cols // spec.restart_columns):
            for step in range(steps_per_block):
                bit_offset = block * block_bits + (steps_per_block - 1 - step) * spec.code_bits
                states[row_index, block * steps_per_block + step] = _manual_bit_field(
                    row, bit_offset, spec.window_bits
                )
    levels = np.asarray(_states_to_levels(jnp.asarray(states)), dtype=np.float64).reshape(packed_tape.shape[0], cols)
    return levels * _codebook_scale() * scales[:, None]


def test_trellis_state_levels_match_uzu_materialized_table() -> None:
    levels = _states_to_levels(jnp.arange(1 << 16, dtype=jnp.uint32))

    assert int(jnp.sum(levels.astype(jnp.int32))) == _MATERIALIZED_LEVEL_SUM
    for state, expected in _MATERIALIZED_LEVELS.items():
        assert tuple(int(level) for level in levels[state]) == expected


def test_trellis_level_table_is_unit_variance_after_scaling() -> None:
    levels = np.array(_level_table(), dtype=np.float64)

    assert levels.shape == (256,)
    assert levels.min() == -54
    assert levels.max() == 55
    assert len(set(_level_table())) == 74
    assert np.mean(np.square(levels * _codebook_scale())) == pytest.approx(1.0, abs=1e-6)


def test_trellis_tape_layout_matches_uzu_golden_tape() -> None:
    spec = TrellisSpec(bits=3, window_bits=32, restart_columns=68)
    layout = _TapeLayout(spec, cols=68)
    golden_tape = jnp.asarray(np.frombuffer(bytes.fromhex(_GOLDEN_TAPE), dtype=np.uint8))[None]
    golden_states = jnp.asarray(np.array(_GOLDEN_STATES, dtype=np.uint32))[None]

    states = _tape_to_states(golden_tape, layout)
    packed_tape = _states_to_tape(golden_states, layout)

    assert layout.row_bytes == len(_GOLDEN_TAPE) // 2
    assert jnp.array_equal(states, golden_states)
    assert jnp.array_equal(packed_tape, golden_tape)


@pytest.mark.parametrize(
    ("bits", "window_bits", "restart_columns", "cols"),
    [
        (2, 16, 64, 256),
        (3, 16, 64, 128),
        (1, 8, 8, 32),
        (2, 12, 8, 24),
        (4, 32, 16, 64),
    ],
)
def test_trellis_tape_roundtrips_states_that_follow_the_recurrence(
    bits: Literal[1, 2, 3, 4],
    window_bits: int,
    restart_columns: int,
    cols: int,
) -> None:
    spec = TrellisSpec(bits=bits, window_bits=window_bits, restart_columns=restart_columns)
    layout = _TapeLayout(spec, cols)
    generator = np.random.default_rng(0)
    headers = generator.integers(0, 1 << window_bits, size=(3, layout.blocks), dtype=np.uint64)
    codes = generator.integers(0, 1 << (4 * bits), size=(3, layout.blocks, spec.steps_per_block - 1), dtype=np.uint64)
    states = jnp.asarray(_recurrence_states(spec, headers, codes))

    packed_tape = _states_to_tape(states, layout)

    assert packed_tape.shape == (3, layout.row_bytes)
    assert jnp.array_equal(_tape_to_states(packed_tape, layout), states)


def test_trellis_viterbi_matches_brute_force_search() -> None:
    spec = TrellisSpec(bits=1, window_bits=8, restart_columns=8)
    weights = _logical_weights()
    codebook = np.asarray(_codebook(spec.window_bits), dtype=np.float64)
    headers = np.arange(1 << spec.window_bits)
    successors = ((headers[:, None] << 4) & 0xFF) | np.arange(16)[None, :]
    path_codewords = np.concatenate(
        [np.broadcast_to(codebook[headers][:, None, :], (256, 16, 4)), codebook[successors]],
        axis=-1,
    )

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    targets = np.asarray(weights / _search_scales(weights)[:, None], dtype=np.float64)
    rows, cols = weights.shape
    layout = _TapeLayout(spec, cols)
    fitted_codewords = codebook[np.asarray(_tape_to_states(matrix.packed_tape, layout))].reshape(rows, cols)
    fitted_costs = np.sum(np.square(targets - fitted_codewords), axis=-1)
    for row in range(rows):
        block_costs = [
            np.min(np.sum(np.square(targets[row, block * 8 : (block + 1) * 8] - path_codewords), axis=-1))
            for block in range(layout.blocks)
        ]
        assert fitted_costs[row] == pytest.approx(sum(block_costs), rel=1e-5)


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_trellis_compress_and_decompress_match_manual_decode(layout: Layout) -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8, layout=layout)
    weights = _logical_weights()[:8]

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    *_, cols = matrix.shape
    reference = _manual_decode(spec, np.asarray(matrix.packed_tape), np.asarray(matrix.scales), cols)
    assert matrix.shape == layout.weight_shape((), *weights.shape)
    assert_close_arrays(
        result=matrix.decompress(),
        reference=layout.to_output_input(jnp.asarray(reference, dtype=jnp.float32)),
    )


@pytest.mark.parametrize(("bits", "window_bits"), [(1, 8), (2, 12), (3, 16), (4, 20)])
def test_trellis_fitted_states_follow_the_recurrence(
    bits: Literal[1, 2, 3, 4],
    window_bits: int,
) -> None:
    spec = TrellisSpec(bits=bits, window_bits=window_bits, restart_columns=16)
    layout = _TapeLayout(spec, cols=32)
    weights = _gaussian_weights(4, 32)

    states = _weights_to_states(weights, layout)

    blocks = np.asarray(states, dtype=np.uint64).reshape(4, layout.blocks, spec.steps_per_block)
    headers = blocks[..., 0]
    codes = blocks[..., 1:] & spec.code_mask
    assert np.array_equal(blocks.reshape(4, -1), _recurrence_states(spec, headers, codes).astype(np.uint64))


def test_trellis_viterbi_chunking_matches_single_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=16)
    layout = _TapeLayout(spec, cols=32)
    weights = _gaussian_weights(8, 32)
    full_states = _weights_to_states(weights, layout)
    segment_bytes = 4 * (1 << spec.window_bits) + (1 << (spec.window_bits - spec.code_bits)) * spec.steps_per_block
    monkeypatch.setattr("lalamo.compressed.trellis._MAX_CHUNK_BYTES", 3 * segment_bytes)

    chunked_states = _weights_to_states(weights, layout)

    assert jnp.array_equal(chunked_states, full_states)


@pytest.mark.parametrize("bits", [2, 3])
def test_trellis_quantization_error_matches_distortion_estimate(bits: Literal[2, 3]) -> None:
    spec = TrellisSpec(bits=bits, window_bits=16, restart_columns=64)
    weights = _gaussian_weights(16, 256)

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    empirical_mse = float(jnp.mean(jnp.square(weights - matrix.decompress())))
    assert empirical_mse == pytest.approx(spec.distortion, rel=0.1)


def test_trellis_distortion_estimates_are_ordered_by_bits_and_by_restart_columns() -> None:
    restarts = (16, 32, 64, 128)
    distortions = {
        (bits, restart): TrellisSpec(bits=bits, window_bits=16, restart_columns=restart).distortion
        for bits in (1, 2, 3, 4)
        for restart in restarts
    }

    for restart in restarts:
        by_bits = [distortions[bits, restart] for bits in (1, 2, 3, 4)]
        assert by_bits == sorted(by_bits, reverse=True)
        assert all(0 < distortion < 1 for distortion in by_bits)
    for bits in (1, 2, 3, 4):
        by_restart = [distortions[bits, restart] for restart in restarts]
        if 4 * bits < 16:
            assert by_restart == sorted(by_restart)
        else:
            assert max(by_restart) == pytest.approx(min(by_restart), rel=0.02)


def test_trellis_all_zero_rows_decompress_to_zeros_with_finite_scales() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    weights = _logical_weights().at[3].set(0)

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    decompressed = compressed_common.host_decompressed(matrix)
    assert bool(jnp.all(jnp.isfinite(matrix.scales)))
    assert_close_arrays(result=decompressed[3], reference=jnp.zeros(16, dtype=jnp.float32))
    reference = spec.compress(weights[4:], sharding_config=make_test_sharding_config()).decompress()
    assert_close_arrays(result=decompressed[4:], reference=reference)


@pytest.mark.parametrize(
    ("layout", "tape_axes", "scale_axes"),
    [
        (Layout.OUTPUT_INPUT, (LogicalAxis.MATRIX, None), (LogicalAxis.MATRIX,)),
        (Layout.INPUT_OUTPUT, (None, None), (None,)),
    ],
)
def test_trellis_compress_shards_tape_and_scales_along_rows(
    layout: Layout,
    tape_axes: tuple[LogicalAxis | None, ...],
    scale_axes: tuple[LogicalAxis | None, ...],
) -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8, layout=layout)

    matrix = spec.compress(_logical_weights(), sharding_config=make_test_sharding_config())

    assert matrix.packed_tape.sharding == make_sharding(tape_axes)
    assert matrix.scales.sharding == make_sharding(scale_axes)


def test_trellis_compress_keeps_weight_dtype_for_scales_and_decompression() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    weights = _logical_weights().astype(jnp.bfloat16)

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    assert matrix.scales.dtype == jnp.bfloat16
    assert matrix.decompress().dtype == jnp.bfloat16
    assert matrix.astype(jnp.float32).decompress().dtype == jnp.float32


def test_trellis_decompress_rounds_once_after_the_float32_scale_product() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    matrix = spec.compress(_logical_weights().astype(jnp.bfloat16), sharding_config=make_test_sharding_config())

    decompressed = matrix.decompress()

    reference = matrix.astype(jnp.float32).decompress().astype(jnp.bfloat16)
    assert decompressed.dtype == jnp.bfloat16
    assert jnp.array_equal(decompressed, reference)


def test_trellis_dummy_template_loads_a_window_too_wide_to_search() -> None:
    spec = TrellisSpec(bits=3, window_bits=32, restart_columns=68)
    golden_row = np.frombuffer(bytes.fromhex(_GOLDEN_TAPE), dtype=np.uint8)
    tape = jnp.asarray(np.stack([golden_row, golden_row]))
    original = spec.from_packed_parameters(
        packed_tape=tape,
        scales=jnp.array([1.0, -0.5], dtype=jnp.float32),
        cols=68,
        sharding_config=make_test_sharding_config(),
    )
    template = spec.compress(
        dummy_array((2, 68), jnp.float32, make_sharding((None, None))),
        sharding_config=make_test_sharding_config(),
    )

    restored = template.load_exported(original.export())

    assert template.packed_tape.shape == tape.shape
    assert template.decompress().shape == (2, 68)
    assert jnp.array_equal(restored.packed_tape, tape)
    assert_close_arrays(result=restored.decompress(), reference=original.decompress())


def _positive_definite_block(size: int, seed: int) -> jax.Array:
    factor = jax.random.normal(jax.random.key(seed), (size, size), dtype=jnp.float32)
    return factor @ factor.T + jnp.identity(size, dtype=jnp.float32) * size


def _input_quadratic_objective(weights: jax.Array, decompressed: jax.Array, input_block: jax.Array) -> float:
    residual = np.asarray(weights, dtype=np.float64) - np.asarray(decompressed, dtype=np.float64)
    return float(np.einsum("oi,ij,oj->", residual, np.asarray(input_block, dtype=np.float64), residual))


def test_trellis_quantize_block_returns_unit_scale_codewords() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    blocks = _gaussian_weights(6, 8)[:, None, :]

    codewords = spec.quantize_block(blocks, sharding_config=make_test_sharding_config())

    states = _weights_to_states(blocks[:, 0, :], _TapeLayout(spec, cols=8))
    assert_close_arrays(result=codewords[:, 0, :], reference=_states_to_codewords(states))


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_trellis_identity_preconditioner_matches_the_plain_fit(layout: Layout) -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8, layout=layout)
    weights = _gaussian_weights(16, 16)
    preconditioner = Preconditioner.init(
        input_block=jnp.identity(16, dtype=jnp.float32),
        output_block=jnp.identity(16, dtype=jnp.float32),
    )

    plain = spec.compress(weights, sharding_config=make_test_sharding_config())
    preconditioned = spec.compress(weights, preconditioner=preconditioner, sharding_config=make_test_sharding_config())

    assert np.array_equal(np.asarray(preconditioned.packed_tape), np.asarray(plain.packed_tape))
    assert_close_arrays(result=preconditioned.scales, reference=plain.scales)


def test_trellis_input_preconditioner_lowers_the_input_weighted_error() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    weights = _gaussian_weights(16, 32)
    input_block = _positive_definite_block(32, seed=7)
    preconditioner = Preconditioner.init(input_block=input_block)

    plain = spec.compress(weights, sharding_config=make_test_sharding_config())
    preconditioned = spec.compress(weights, preconditioner=preconditioner, sharding_config=make_test_sharding_config())

    plain_objective = _input_quadratic_objective(weights, plain.decompress(), input_block)
    preconditioned_objective = _input_quadratic_objective(weights, preconditioned.decompress(), input_block)
    assert preconditioned_objective < plain_objective
    assert preconditioned.packed_tape.shape == plain.packed_tape.shape


def test_trellis_load_exported_rejects_column_count_mismatch() -> None:
    spec = TrellisSpec(bits=1, window_bits=4, restart_columns=4)
    narrow = spec.compress(jnp.ones((2, 4), dtype=jnp.float32), sharding_config=make_test_sharding_config())
    wide = spec.compress(jnp.ones((2, 8), dtype=jnp.float32), sharding_config=make_test_sharding_config())
    assert narrow.packed_tape.shape == wide.packed_tape.shape

    with pytest.raises(ValueError, match="cols mismatch"):
        wide.load_exported(narrow.export())


def test_trellis_transposed_dot_matches_transposed_decompressed_weights() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    matrix = spec.compress(_logical_weights(), sharding_config=make_test_sharding_config())
    vector = jnp.linspace(-1, 1, 16, dtype=jnp.float32)
    keychain = Keychain.init(0, sharding_config=make_test_sharding_config())

    result = matrix.dot(vector, keychain=keychain, transposed=True)

    assert_close_arrays(result=result, reference=matrix.decompress().T @ vector)


def test_trellis_lookup_embedding_supports_batched_indices(fake_mesh: Mesh) -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8, layout=Layout.INPUT_OUTPUT)
    matrix = spec.compress(_logical_weights(), sharding_config=make_test_sharding_config())
    row_index = jax.device_put(jnp.array([2, 5, 2, 7]), make_sharding((LogicalAxis.BATCH,)))
    table = matrix.decompress().T

    result = matrix.lookup_embedding(row_index, keychain=Keychain.init(0, sharding_config=make_test_sharding_config()))

    assert_close_arrays(result=result, reference=table[np.asarray(row_index)])
    assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((LogicalAxis.BATCH, None))


def test_trellis_scales_are_least_squares_optimal_for_the_fitted_codes() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=16)
    weights = _gaussian_weights(8, 64)

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    codewords = matrix.decompress() / matrix.scales[:, None]
    expected_scales = jnp.sum(weights * codewords, axis=-1) / jnp.sum(jnp.square(codewords), axis=-1)
    assert_close_arrays(result=matrix.scales, reference=expected_scales)


def test_trellis_dot_gradient_flows_to_scales_only() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    matrix = spec.compress(_logical_weights(), sharding_config=make_test_sharding_config())
    vector = jnp.linspace(-1, 1, 16, dtype=jnp.float32)
    keychain = Keychain.init(0, sharding_config=make_test_sharding_config())

    def objective(scales: jax.Array) -> jax.Array:
        return jnp.sum(eqx.tree_at(lambda m: m.scales, matrix, scales).dot(vector, keychain=keychain))

    gradient = jax.grad(objective)(matrix.scales)

    codewords = matrix.decompress() / matrix.scales[:, None]
    assert_close_arrays(result=gradient, reference=codewords @ vector)
    inexact_leaves = jax.tree_util.tree_leaves(eqx.filter(matrix, eqx.is_inexact_array))
    assert len(inexact_leaves) == 1
    assert inexact_leaves[0] is matrix.scales


def test_trellis_compress_mixture_matches_per_component_compress() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    weights = _logical_weights(2) * jnp.array([1.0, -0.5], dtype=jnp.float32)[:, None, None]

    matrix = spec.compress(weights, sharding_config=make_test_sharding_config())

    reference = jnp.stack(
        [spec.compress(weights[index], sharding_config=make_test_sharding_config()).decompress() for index in range(2)]
    )
    assert matrix.shape == weights.shape
    assert matrix.packed_tape.shape == (2, 16, _TapeLayout(spec, cols=16).row_bytes)
    assert matrix.scales.shape == (2, 16)
    assert_close_arrays(result=matrix.decompress(), reference=reference)


def test_trellis_switch_sharding_config_keeps_the_tape() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    matrix = spec.compress(_logical_weights(), sharding_config=make_test_sharding_config())

    switched = matrix.switch_sharding_config(ShardingConfig.replicated())

    assert matrix.switch_sharding_config(make_test_sharding_config()) is matrix
    assert switched.spec == spec
    assert switched.cols == matrix.cols
    assert np.array_equal(np.asarray(switched.packed_tape), np.asarray(matrix.packed_tape))
    assert np.array_equal(np.asarray(switched.scales), np.asarray(matrix.scales))


@pytest.mark.parametrize(
    "spec",
    [
        TrellisSpec(bits=2, window_bits=16, restart_columns=64),
        TrellisSpec(bits=3, window_bits=16, restart_columns=64),
        TrellisSpec(bits=1, window_bits=8, restart_columns=16),
        TrellisSpec(bits=4, window_bits=16, restart_columns=32),
    ],
)
def test_trellis_rate_matches_tape_bytes_plus_one_scale_per_row(spec: TrellisSpec) -> None:
    rows, cols = 8, 256

    matrix = spec.from_packed_parameters(
        packed_tape=jnp.zeros((rows, _TapeLayout(spec, cols).row_bytes), dtype=jnp.uint8),
        scales=jnp.ones((rows,), dtype=jnp.float32),
        cols=cols,
        sharding_config=make_test_sharding_config(),
    )

    byte_count = sum(array.size * array.dtype.itemsize for array in matrix.export().arrays.values())
    scale_bits_per_weight = 8 * jnp.dtype(matrix.scales.dtype).itemsize / cols
    assert 8 * byte_count / (rows * cols) == pytest.approx(spec.rate + scale_bits_per_weight)


def test_trellis_spec_json_roundtrip() -> None:
    spec = TrellisSpec(bits=3, window_bits=16, restart_columns=64, layout=Layout.INPUT_OUTPUT)

    assert WeightMatrixSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize(
    ("bits", "window_bits", "restart_columns", "message"),
    [
        (2, 4, 64, "between 8 and 32"),
        (2, 33, 64, "between 8 and 32"),
        (4, 12, 64, "between 16 and 32"),
        (2, 16, 6, "positive multiple of 4"),
        (2, 16, 0, "positive multiple of 4"),
    ],
)
def test_trellis_spec_rejects_invalid_configurations(
    bits: int,
    window_bits: int,
    restart_columns: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrellisSpec(bits=bits, window_bits=window_bits, restart_columns=restart_columns)  # type: ignore[arg-type]


def test_trellis_compress_rejects_columns_not_divisible_by_restart_columns() -> None:
    weights = jnp.ones((4, 24), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        TrellisSpec(bits=2, window_bits=12, restart_columns=16).compress(
            weights, sharding_config=make_test_sharding_config()
        )


def test_trellis_compress_rejects_windows_too_wide_to_search() -> None:
    weights = jnp.ones((2, 68), dtype=jnp.float32)

    with pytest.raises(ValueError, match="at most 24"):
        TrellisSpec(bits=3, window_bits=32, restart_columns=68).compress(
            weights, sharding_config=make_test_sharding_config()
        )


def test_trellis_from_packed_parameters_rejects_tape_rows_of_the_wrong_length() -> None:
    spec = TrellisSpec(bits=2, window_bits=16, restart_columns=64)

    with pytest.raises(ValueError, match="Expected 34-byte tape rows"):
        spec.from_packed_parameters(
            packed_tape=jnp.zeros((4, 20), dtype=jnp.uint8),
            scales=jnp.ones((4,), dtype=jnp.float32),
            cols=128,
            sharding_config=make_test_sharding_config(),
        )


def test_trellis_decompress_multiplies_the_level_by_the_folded_scale() -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8)
    matrix = spec.compress(_logical_weights(), sharding_config=make_test_sharding_config())

    decompressed = np.asarray(matrix.decompress())

    states = _tape_to_states(matrix.packed_tape, _TapeLayout(spec, matrix.cols))
    levels = np.asarray(_states_to_levels(states), dtype=np.float32).reshape(matrix.shape)
    folded_scales = np.asarray(matrix.scales, dtype=np.float32) * np.float32(_codebook_scale())
    assert np.array_equal(decompressed, levels * folded_scales[:, None])


@pytest.mark.parametrize("layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
@pytest.mark.parametrize("leading_dims", [(), (2,)])
def test_trellis_compress_sharded_and_unsharded_agree(layout: Layout, leading_dims: tuple[int, ...]) -> None:
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8, layout=layout)
    weights = _logical_weights(*leading_dims)

    sharded = spec.compress(weights, sharding_config=make_test_sharding_config())
    unsharded = spec.compress(weights, sharding_config=make_test_sharding_config(), is_sharded=False)

    assert np.array_equal(np.asarray(sharded.packed_tape), np.asarray(unsharded.packed_tape))
    assert np.array_equal(np.asarray(sharded.scales), np.asarray(unsharded.scales))


def test_trellis_inside_hybrid_builds_templates_and_reshards_without_a_search() -> None:
    trellis_spec = TrellisSpec(bits=3, window_bits=32, restart_columns=68)
    spec = HybridSpec(quantization_spec=trellis_spec, adapter_spec=None, incoherence_block_size=None)
    golden_row = np.frombuffer(bytes.fromhex(_GOLDEN_TAPE), dtype=np.uint8)
    quantized = trellis_spec.from_packed_parameters(
        packed_tape=jnp.asarray(np.stack([golden_row, golden_row])),
        scales=jnp.array([1.0, -0.5], dtype=jnp.float32),
        cols=68,
        sharding_config=make_test_sharding_config(),
    )
    hybrid = HybridMatrix(
        spec=spec,
        sharding_config=make_test_sharding_config(),
        is_sharded=True,
        quantized=quantized,
        adapter=None,
        incoherence_signs=None,
    )
    template = spec.compress(
        dummy_array((2, 68), jnp.float32, make_sharding((None, None))),
        sharding_config=make_test_sharding_config(),
    )

    restored = template.load_exported(hybrid.export())
    resharded = restored.switch_sharding_config(ShardingConfig.replicated())

    assert isinstance(resharded, HybridMatrix)
    assert isinstance(resharded.quantized, TrellisMatrix)
    assert np.array_equal(np.asarray(resharded.quantized.packed_tape), np.asarray(quantized.packed_tape))
    assert np.array_equal(np.asarray(resharded.quantized.scales), np.asarray(quantized.scales))
    assert_close_arrays(result=restored.decompress(), reference=quantized.decompress())


def test_trellis_from_packed_parameters_rejects_scales_that_are_not_one_per_row() -> None:
    spec = TrellisSpec(bits=2, window_bits=16, restart_columns=64)

    with pytest.raises((TypeError, ValueError), match="row"):
        spec.from_packed_parameters(
            packed_tape=jnp.zeros((8, 17), dtype=jnp.uint8),
            scales=jnp.ones((1,), dtype=jnp.float32),
            cols=64,
            sharding_config=make_test_sharding_config(),
        )


def test_trellis_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    weights = _logical_weights()
    spec = TrellisSpec(bits=2, window_bits=12, restart_columns=8, layout=Layout.INPUT_OUTPUT)
    saved_sharding = make_sharding((LogicalAxis.MATRIX, None))
    assert saved_sharding is not None
    original = spec.compress(weights, sharding_config=make_test_sharding_config())
    reference = original.decompress()
    original = _put_on_sharding(original, saved_sharding, make_sharding((LogicalAxis.MATRIX,)))
    template = spec.compress(
        dummy_array(weights.shape, weights.dtype, make_sharding((None, None))),
        sharding_config=make_test_sharding_config(),
    )

    restored = template.load_exported(original.export())

    assert set(original.export().arrays) == {"weights", "scales"}
    assert template.decompress().shape == weights.shape
    assert restored.spec == spec
    assert isinstance(restored, TrellisMatrix)
    assert_close_arrays(result=restored.decompress(), reference=reference)
    del fake_mesh
    assert not is_sharded(restored.scales.sharding)
    assert not is_sharded(restored.packed_tape.sharding)
    assert restored.packed_tape.sharding != saved_sharding


def test_trellis_load_exported_rejects_spec_mismatch() -> None:
    weights = _logical_weights()
    matrix = TrellisSpec(bits=2, window_bits=12, restart_columns=8).compress(
        weights, sharding_config=make_test_sharding_config()
    )
    template = TrellisSpec(bits=2, window_bits=16, restart_columns=8).compress(
        weights, sharding_config=make_test_sharding_config()
    )

    with pytest.raises(ValueError, match="spec mismatch"):
        template.load_exported(matrix.export())
