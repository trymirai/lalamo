from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, cached_property, partial
from math import ceil, sqrt
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.lax import DotAlgorithmPreset
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int8, Key, UInt, UInt8, UInt32

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import (
    dummy_array,
    is_dummy_array,
    preserve_first_input_sharding,
    supports_dummy_arrays,
)
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, is_sharded, sharding_of, with_sharding
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

from .data.distortion import distortion_estimate
from .quantized_spec import QuantizedSpec
from .utils.yaqa import yaqa_round_blockwise

__all__ = [
    "TrellisMatrix",
    "TrellisSpec",
]

WEIGHTS_PER_STATE = 4
MAX_SEARCH_WINDOW_BITS = 24
MAX_CHUNK_BYTES = 134_217_728
# Low 32 bits of SplitMix64 at 0 (forced odd) and 1 with seed 1234, as uzu's trellis_format.rs derives them.
HASH_MULTIPLIER = 3486300223
HASH_INCREMENT = 1481329315


def level_of_byte(byte: int) -> int:
    pairs = sum((byte >> shift) & 3 for shift in range(0, 8, 2))
    return 8 * pairs + ((3 * (byte & 15)) & 15) - 54


@cache
def level_table() -> tuple[int, ...]:
    return tuple(level_of_byte(byte) for byte in range(256))


@cache
def codebook_scale() -> float:
    levels = level_table()
    return 1 / sqrt(sum(level * level for level in levels) / len(levels))


def state_hashes(states: UInt32[Array, "..."]) -> UInt32[Array, "..."]:
    mixed = states * jnp.uint32(HASH_MULTIPLIER) + jnp.uint32(HASH_INCREMENT)
    mixed = mixed ^ (mixed >> jnp.uint32(16))
    mixed = mixed * jnp.uint32(0x85EBCA6B)
    return mixed ^ (mixed >> jnp.uint32(16))


def states_to_levels(states: UInt32[Array, "..."]) -> Int8[Array, "... 4"]:
    hashes = state_hashes(states)
    pairs = (hashes & jnp.uint32(0x33333333)) + ((hashes >> jnp.uint32(2)) & jnp.uint32(0x33333333))
    pairs = (pairs & jnp.uint32(0x0F0F0F0F)) + ((pairs >> jnp.uint32(4)) & jnp.uint32(0x0F0F0F0F))
    dither = (jnp.uint32(3) * (hashes & jnp.uint32(0x0F0F0F0F))) & jnp.uint32(0x0F0F0F0F)
    packed = ((pairs << jnp.uint32(3)) + dither + jnp.uint32(0x4A4A4A4A)) ^ jnp.uint32(0x80808080)
    return jax.lax.bitcast_convert_type(packed, jnp.int8)


def states_to_codewords(states: UInt32[Array, "... steps"]) -> Float[Array, "... cols"]:
    levels = rearrange(states_to_levels(states), "... steps weights -> ... (steps weights)")
    return levels.astype(jnp.float32) * jnp.float32(codebook_scale())


def window_codebook(window_bits: int) -> Float[Array, "states 4"]:
    states = jnp.arange(1 << window_bits, dtype=jnp.uint32)
    return states_to_levels(states).astype(jnp.float32) * jnp.float32(codebook_scale())


@dataclass(frozen=True)
class TapeLayout:
    spec: "TrellisSpec"
    cols: int

    def __post_init__(self) -> None:
        if self.cols % self.spec.restart_columns != 0:
            raise ValueError(f"cols={self.cols} must be divisible by restart_columns={self.spec.restart_columns}")

    @property
    def blocks(self) -> int:
        return self.cols // self.spec.restart_columns

    @property
    def steps(self) -> int:
        return self.blocks * self.spec.steps_per_block

    @property
    def row_bytes(self) -> int:
        return ceil(self.blocks * self.spec.block_bits / 8)

    @property
    def word_count(self) -> int:
        payload_words = ceil(self.blocks * self.spec.block_bits / 32)
        return payload_words + 1

    def window_bit_offsets(self) -> UInt32[np.ndarray, " steps"]:
        block_starts = np.arange(self.blocks) * self.spec.block_bits
        step_offsets = (self.spec.steps_per_block - 1 - np.arange(self.spec.steps_per_block)) * self.spec.code_bits
        return (block_starts[:, None] + step_offsets[None, :]).reshape(-1).astype(np.uint32)

    def field_bit_masks(self) -> UInt32[np.ndarray, " steps"]:
        is_header = np.arange(self.steps) % self.spec.steps_per_block == 0
        return np.where(is_header, self.spec.state_mask, self.spec.code_mask).astype(np.uint32)


def bytes_to_words(packed_tape: UInt8[Array, "... packed_cols"], word_count: int) -> UInt32[Array, "... words"]:
    *_, row_bytes = packed_tape.shape
    padding = [(0, 0)] * (packed_tape.ndim - 1) + [(0, 4 * word_count - row_bytes)]
    grouped = rearrange(jnp.pad(packed_tape, padding), "... (words bytes) -> ... words bytes", bytes=4)
    return jax.lax.bitcast_convert_type(grouped, jnp.uint32)


def words_to_bytes(words: UInt32[Array, "... words"], row_bytes: int) -> UInt8[Array, "... packed_cols"]:
    grouped = jax.lax.bitcast_convert_type(words, jnp.uint8)
    return rearrange(grouped, "... words bytes -> ... (words bytes)")[..., :row_bytes]


def tape_to_states(packed_tape: UInt8[Array, "... packed_cols"], layout: TapeLayout) -> UInt32[Array, "... steps"]:
    words = bytes_to_words(packed_tape, layout.word_count)
    offsets = layout.window_bit_offsets()
    word_indices = jnp.asarray(offsets // 32, dtype=jnp.int32)
    shifts = jnp.asarray(offsets % 32, dtype=jnp.uint32)
    word_pairs = words[..., jnp.stack([word_indices, word_indices + 1], axis=-1)]
    low = word_pairs[..., 0] >> shifts
    high = jnp.where(shifts == 0, jnp.uint32(0), word_pairs[..., 1] << (jnp.uint32(32) - shifts))
    return (low | high) & jnp.uint32(layout.spec.state_mask)


def states_to_tape(states: UInt32[Array, "... steps"], layout: TapeLayout) -> UInt8[Array, "... packed_cols"]:
    *leading_dims, _ = states.shape
    offsets = layout.window_bit_offsets()
    word_indices = jnp.asarray(offsets // 32, dtype=jnp.int32)
    shifts = jnp.asarray(offsets % 32, dtype=jnp.uint32)
    values = states & jnp.asarray(layout.field_bit_masks())
    low = values << shifts
    high = jnp.where(shifts == 0, jnp.uint32(0), values >> (jnp.uint32(32) - shifts))
    words = jnp.zeros((*leading_dims, layout.word_count), dtype=jnp.uint32)
    words = words.at[..., word_indices].add(low).at[..., word_indices + 1].add(high)
    return words_to_bytes(words, layout.row_bytes)


def backpointer_dtype(code_bits: int) -> DTypeLike:
    if code_bits <= 8:
        return jnp.uint8
    return jnp.uint16


def viterbi_segment_states(
    targets: Float[Array, "steps 4"],
    *,
    codebook: Float[Array, "states 4"],
    window_bits: int,
    code_bits: int,
) -> UInt32[Array, " steps"]:
    num_predecessors = 1 << code_bits
    num_suffixes = 1 << (window_bits - code_bits)
    codeword_norms = jnp.sum(jnp.square(codebook), axis=-1)
    pointer_dtype = backpointer_dtype(code_bits)

    def branch_costs(target: Float[Array, " 4"]) -> Float[Array, " states"]:
        with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
            return codeword_norms - jnp.float32(2) * (codebook @ target)

    # The predecessors of a state are the states whose low window_bits - code_bits bits equal its high bits,
    # predecessor = high * num_suffixes + (state >> code_bits), so one backpointer per suffix covers all states.
    def forward(
        costs: Float[Array, " states"],
        target: Float[Array, " 4"],
    ) -> tuple[Float[Array, " states"], UInt[Array, " suffixes"]]:
        grouped = rearrange(costs, "(predecessors suffixes) -> predecessors suffixes", suffixes=num_suffixes)
        best_predecessors = jnp.argmin(grouped, axis=0).astype(pointer_dtype)
        costs = branch_costs(target) + jnp.repeat(jnp.min(grouped, axis=0), num_predecessors)
        return costs, best_predecessors

    def backward(
        state: UInt32[Array, ""],
        backpointers: UInt[Array, " suffixes"],
    ) -> tuple[UInt32[Array, ""], UInt32[Array, ""]]:
        suffix = state >> jnp.uint32(code_bits)
        predecessor = backpointers[suffix].astype(jnp.uint32) << jnp.uint32(window_bits - code_bits)
        return predecessor | suffix, state

    first_target, later_targets = targets[0], targets[1:]
    final_costs, backpointers = jax.lax.scan(forward, branch_costs(first_target), later_targets)
    last_state = jnp.argmin(final_costs).astype(jnp.uint32)
    first_state, later_states = jax.lax.scan(backward, last_state, backpointers, reverse=True)
    return jnp.concatenate([first_state[None], later_states])


def weights_to_states(weights: Float[Array, "... cols"], layout: TapeLayout) -> UInt32[Array, "... steps"]:
    *leading_dims, cols = weights.shape
    segments = rearrange(
        weights.reshape(-1, cols),
        "rows (blocks steps weights) -> (rows blocks) steps weights",
        steps=layout.spec.steps_per_block,
        weights=WEIGHTS_PER_STATE,
    )
    codebook = window_codebook(layout.spec.window_bits)
    num_states, _ = codebook.shape
    backpointer_bytes = jnp.dtype(backpointer_dtype(layout.spec.code_bits)).itemsize
    segment_bytes = (
        4 * num_states + backpointer_bytes * (num_states >> layout.spec.code_bits) * layout.spec.steps_per_block
    )
    fit_segment = partial(
        viterbi_segment_states,
        codebook=codebook,
        window_bits=layout.spec.window_bits,
        code_bits=layout.spec.code_bits,
    )
    states = jax.lax.map(fit_segment, segments, batch_size=max(1, MAX_CHUNK_BYTES // segment_bytes))
    return states.reshape(*leading_dims, layout.steps)


def shard_rows_only(weights: Float[Array, "... rows cols"]) -> Float[Array, "... rows cols"]:
    sharding = sharding_of(weights)
    if not is_sharded(sharding):
        return weights
    *row_axes, cols_axis = tuple(sharding.spec) + (None,) * (weights.ndim - len(sharding.spec))
    *_, rows, _ = weights.shape
    if cols_axis is not None and row_axes[-1] is None and rows % sharding.mesh.shape[cols_axis] == 0:
        row_axes[-1] = cols_axis
    return with_sharding(weights, NamedSharding(sharding.mesh, PartitionSpec(*row_axes, None)))


def map_over_rows[ResultT](function: Callable[[Array], ResultT], rows: Array) -> ResultT:
    sharding = sharding_of(rows)
    if not is_sharded(sharding):
        return function(rows)
    return jax.shard_map(function, mesh=sharding.mesh, in_specs=sharding.spec, out_specs=sharding.spec)(rows)


def search_scales(weights: Float[Array, "... cols"]) -> Float[Array, "..."]:
    scales = jnp.sqrt(jnp.mean(jnp.square(weights), axis=-1))
    return jnp.where(scales == 0, 1, scales)


class PackedParameters(NamedTuple):
    packed_tape: UInt8[Array, "... packed_cols"]
    scales: Float[Array, "..."]


def scaled_preconditioner(
    preconditioner: Preconditioner,
    scales: Float[Array, "*components rows"],
    layout: Layout,
) -> Preconditioner:
    input_block = preconditioner.input_block
    output_block = preconditioner.output_block
    outer = scales[..., :, None] * scales[..., None, :]
    if layout == Layout.INPUT_OUTPUT and input_block is not None:
        input_block = input_block * outer.astype(input_block.dtype)
    if layout == Layout.OUTPUT_INPUT and output_block is not None:
        output_block = output_block * outer.astype(output_block.dtype)
    return Preconditioner.init(input_block=input_block, output_block=output_block)


def row_metric(preconditioner: Preconditioner, layout: Layout) -> Float[Array, "*components cols cols"] | None:
    if layout == Layout.INPUT_OUTPUT:
        return preconditioner.output_block
    return preconditioner.input_block


def least_squares_scales(
    weights: Float[Array, "... cols"],
    codewords: Float[Array, "... cols"],
    metric: Float[Array, "... cols cols"] | None,
) -> Float[Array, "..."]:
    weighted_codewords = codewords
    if metric is not None:
        with use_dot_algorithm_preset(DotAlgorithmPreset.F32_F32_F32):
            weighted_codewords = jnp.einsum(
                "...rc,...cd->...rd",
                codewords,
                metric.astype(jnp.float32),
                out_sharding=sharding_of(codewords),
            )
    correlations = jnp.sum(weights * weighted_codewords, axis=-1)
    energies = jnp.sum(codewords * weighted_codewords, axis=-1)
    safe_energies = jnp.where(energies == 0, 1, energies)
    return jnp.where(energies == 0, 0, correlations / safe_energies)


def weights_to_packed_parameters(weights: Float[Array, "... cols"], layout: TapeLayout) -> PackedParameters:
    targets = shard_rows_only(weights.astype(jnp.float32))
    normalized_targets = targets / search_scales(targets)[..., None]
    states = map_over_rows(partial(weights_to_states, layout=layout), normalized_targets)
    scales = least_squares_scales(targets, states_to_codewords(states), metric=None)
    packed_tape = map_over_rows(partial(states_to_tape, layout=layout), states)
    return PackedParameters(packed_tape, scales.astype(weights.dtype))


def divides_mesh(sharding: NamedSharding, size: int) -> bool:
    *_, axis = sharding.spec
    return axis is None or size % sharding.mesh.shape[axis] == 0


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def packed_parameters_to_weights(
    packed_tape: UInt8[Array, "... packed_cols"],
    scales: Float[Array, "..."],
    layout: TapeLayout,
    dtype: DTypeLike,
    *,
    states_sharding: NamedSharding,
) -> Float[Array, "... cols"]:
    states = with_sharding(tape_to_states(packed_tape, layout), states_sharding)
    levels = rearrange(states_to_levels(states), "... steps weights -> ... (steps weights)")
    folded_scales = scales.astype(jnp.float32) * jnp.float32(codebook_scale())
    return (levels.astype(jnp.float32) * folded_scales[..., None]).astype(dtype)


@dataclass(frozen=True)
class TrellisSpec(QuantizedSpec):
    bits: Literal[1, 2, 3, 4]
    window_bits: int
    restart_columns: int
    layout: Layout = Layout.OUTPUT_INPUT

    def __post_init__(self) -> None:
        if self.bits not in (1, 2, 3, 4):
            raise ValueError(f"bits must be 1, 2, 3 or 4, got {self.bits}")
        if not self.code_bits <= self.window_bits <= 32:
            raise ValueError(
                f"window_bits must be between {self.code_bits} and 32 for bits={self.bits}, got {self.window_bits}"
            )
        if self.restart_columns <= 0 or self.restart_columns % WEIGHTS_PER_STATE != 0:
            raise ValueError(
                f"restart_columns must be a positive multiple of {WEIGHTS_PER_STATE}, got {self.restart_columns}"
            )

    @property
    def code_bits(self) -> int:
        return self.bits * WEIGHTS_PER_STATE

    @property
    def steps_per_block(self) -> int:
        return self.restart_columns // WEIGHTS_PER_STATE

    @property
    def block_bits(self) -> int:
        return self.window_bits + (self.steps_per_block - 1) * self.code_bits

    @property
    def state_mask(self) -> int:
        return (1 << self.window_bits) - 1

    @property
    def code_mask(self) -> int:
        return (1 << self.code_bits) - 1

    @property
    def input_block_size(self) -> int:
        if self.layout == Layout.INPUT_OUTPUT:
            return 1
        return self.restart_columns

    @property
    def output_block_size(self) -> int:
        if self.layout == Layout.INPUT_OUTPUT:
            return self.restart_columns
        return 1

    def quantize_block(
        self,
        weights: Float[Array, "*blocks out_block_channels in_block_channels"],
        *,
        sharding_config: ShardingConfig,  # noqa: ARG002
    ) -> Float[Array, "*blocks out_block_channels in_block_channels"]:
        expected_shape = (self.output_block_size, self.input_block_size)
        *_, output_block_size, input_block_size = weights.shape
        actual_shape = (output_block_size, input_block_size)
        if actual_shape != expected_shape:
            raise ValueError(f"Expected quantization block shape {expected_shape}, got {actual_shape}")
        rows = self.layout.from_output_input(weights.astype(jnp.float32), sharding=sharding_of(weights))
        states = weights_to_states(rows, TapeLayout(self, self.restart_columns))
        codewords = states_to_codewords(states).reshape(rows.shape)
        return self.layout.to_output_input(codewords).astype(weights.dtype)

    @property
    def rate(self) -> float:
        return self.block_bits / self.restart_columns

    @cached_property
    def distortion(self) -> float:
        return distortion_estimate(
            format_name="trellis",
            bits=self.bits,
            group_size=self.restart_columns,
            window_bits=self.window_bits,
        )

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "TrellisMatrix":
        weight_axes = self.layout.weight_partition(weights.ndim - 2, is_sharded=is_sharded)
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        stored_weights = self.layout.from_output_input(weights, sharding=weight_sharding)
        *leading_dims, cols = stored_weights.shape
        layout = TapeLayout(self, cols)

        if is_dummy_array(weights):
            tape_sharding = sharding_config.make_sharding((None,) * stored_weights.ndim)
            scale_sharding = sharding_config.make_sharding((None,) * len(leading_dims))
            packed_tape = dummy_array((*leading_dims, layout.row_bytes), jnp.uint8, tape_sharding)
            scales = dummy_array(tuple(leading_dims), weights.dtype, scale_sharding)
        else:
            if self.window_bits > MAX_SEARCH_WINDOW_BITS:
                raise ValueError(
                    f"Searching 2**{self.window_bits} states is not supported, "
                    f"compression needs window_bits of at most {MAX_SEARCH_WINDOW_BITS}"
                )
            if preconditioner is None:
                packed_tape, scales = weights_to_packed_parameters(stored_weights, layout)
            else:
                packed_tape, scales = self.preconditioned_packed_parameters(
                    stored_weights,
                    preconditioner,
                    layout,
                    weight_sharding=weight_sharding,
                    sharding_config=sharding_config,
                )

        return self.from_packed_parameters(
            packed_tape=packed_tape,
            scales=scales,
            cols=cols,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
        )

    def preconditioned_packed_parameters(
        self,
        stored_weights: Float[Array, "*components rows cols"],
        preconditioner: Preconditioner,
        layout: TapeLayout,
        *,
        weight_sharding: NamedSharding,
        sharding_config: ShardingConfig,
    ) -> PackedParameters:
        stored_weights = shard_rows_only(stored_weights)
        scales = search_scales(stored_weights.astype(jnp.float32))
        # The sweep replicates its inputs; hand them over replicated so nothing depends on the caller's sharding.
        replicated_scales = with_sharding(scales, sharding_config.make_sharding((None,) * scales.ndim))
        normalized_weights = with_sharding(
            self.layout.to_output_input(stored_weights.astype(jnp.float32) / scales[..., None]),
            sharding_config.make_sharding((None,) * stored_weights.ndim),
        )
        rounded_weights = yaqa_round_blockwise(
            normalized_weights,
            scaled_preconditioner(preconditioner, replicated_scales, self.layout),
            self,
            sharding_config=sharding_config,
        )
        codewords = shard_rows_only(self.layout.from_output_input(rounded_weights, sharding=weight_sharding))
        states = map_over_rows(partial(weights_to_states, layout=layout), codewords)
        targets = with_sharding(stored_weights.astype(jnp.float32), sharding_of(codewords))
        scales = least_squares_scales(targets, codewords, row_metric(preconditioner, self.layout))
        packed_tape = map_over_rows(partial(states_to_tape, layout=layout), states)
        return PackedParameters(packed_tape, scales.astype(stored_weights.dtype))

    def from_packed_parameters(
        self,
        *,
        packed_tape: UInt8[Array, "*components rows packed_cols"],
        scales: Float[Array, "*components rows"],
        cols: int,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "TrellisMatrix":
        layout = TapeLayout(self, cols)
        *row_dims, row_bytes = packed_tape.shape
        if row_bytes != layout.row_bytes:
            raise ValueError(f"Expected {layout.row_bytes}-byte tape rows for cols={cols}, got {row_bytes}")
        if scales.shape != tuple(row_dims):
            raise ValueError(f"Expected one scale per tape row, shape {tuple(row_dims)}, got {scales.shape}")

        *parameter_axes, _ = self.layout.weight_partition(scales.ndim - 1, is_sharded=is_sharded)
        return TrellisMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            cols=cols,
            packed_tape=with_sharding(packed_tape, sharding_config.resolve_sharding((*parameter_axes, None))),
            scales=with_sharding(scales, sharding_config.resolve_sharding(tuple(parameter_axes))),
        )


class TrellisMatrix(WeightMatrix[TrellisSpec]):
    cols: int = field(static=True)
    packed_tape: UInt8[Array, "*components rows packed_cols"]
    scales: Float[Array, "*components rows"] = field(norm=ParameterNorm.L_INF)

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, _ = self.packed_tape.shape
        return (*leading_dims, self.cols)

    @property
    def dtype(self) -> DTypeLike:
        return self.scales.dtype

    def astype(self, dtype: DTypeLike) -> "TrellisMatrix":
        return TrellisMatrix(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            cols=self.cols,
            packed_tape=self.packed_tape,
            scales=self.scales.astype(dtype),
        )

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={"weights": self.packed_tape, "scales": self.scales},
            metadata={"spec": self.spec.to_json(), "cols": self.cols},
        )

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> "TrellisMatrix":
        if prefix is None:
            prefix = ParameterPath()
        loaded_spec = WeightMatrixSpec.from_json(exported_data.metadata[prefix / "spec"])
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")
        loaded_cols = exported_data.metadata[prefix / "cols"]
        if loaded_cols != self.cols:
            raise ValueError(f"WeightMatrix cols mismatch: expected {self.cols}, got {loaded_cols}")
        packed_tape = exported_data.arrays[prefix / "weights"]
        assert packed_tape.dtype == jnp.uint8, f"Trellis tapes are bytes, got {packed_tape.dtype}"
        return TrellisMatrix(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            cols=self.cols,
            packed_tape=load_as(self.packed_tape, packed_tape),
            scales=load_as(self.scales, exported_data.arrays[prefix / "scales"]),
        )

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return self.spec.layout.to_output_input(self.stored_weights_for_forward(self.dtype))

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        weights = self.stored_weights_for_forward(vector.dtype)
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(weights, vector)

    def stored_weights_for_forward(self, dtype: DTypeLike) -> Float[Array, "*components rows cols"]:
        *parameter_axes, cols_axis = self.spec.layout.weight_partition(
            self.scales.ndim - 1,
            is_sharded=self.is_sharded,
        )
        layout = TapeLayout(self.spec, self.cols)
        weight_sharding = self._resolve_sharding((*parameter_axes, cols_axis))
        states_sharding = weight_sharding
        if not divides_mesh(states_sharding, layout.steps):
            states_sharding = self._resolve_sharding((*parameter_axes, None))
        weights = packed_parameters_to_weights(
            self.packed_tape,
            self.scales,
            layout,
            dtype,
            states_sharding=states_sharding,
        )
        return with_sharding(weights, weight_sharding)
