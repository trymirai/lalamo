from dataclasses import dataclass
from functools import cache, cached_property, partial
from math import ceil, sqrt
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int, Int8, Key, UInt, UInt8, UInt32

from lalamo.exportable import ExportResults
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array, is_dummy_array, preserve_first_input_sharding, supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, lookup_sharded_indices, with_sharding
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .data.distortion import distortion_estimate
from .quantized_spec import QuantizedSpec

__all__ = [
    "TrellisMatrix",
    "TrellisSpec",
]

_WEIGHTS_PER_STATE = 4
_CODEBOOK_SEED = 1234
_MAX_SEARCH_WINDOW_BITS = 24
_MAX_STATES_PER_CHUNK = 16_777_216
_MAX_BACKPOINTERS_PER_CHUNK = 67_108_864


class _HashParameters(NamedTuple):
    multiplier: int
    increment: int


def _splitmix64(value: int, seed: int) -> int:
    mask = (1 << 64) - 1
    mixed = (value + seed) & mask
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
    return mixed ^ (mixed >> 31)


@cache
def _hash_parameters() -> _HashParameters:
    mask = (1 << 32) - 1
    return _HashParameters(
        multiplier=(_splitmix64(0, _CODEBOOK_SEED) & mask) | 1,
        increment=_splitmix64(1, _CODEBOOK_SEED) & mask,
    )


@cache
def _level_table() -> tuple[int, ...]:
    levels = []
    for byte in range(256):
        pairs = sum((byte >> shift) & 3 for shift in range(0, 8, 2))
        levels.append(8 * pairs + ((3 * (byte & 15)) & 15) - 54)
    return tuple(levels)


@cache
def _codebook_scale() -> float:
    levels = _level_table()
    return 1 / sqrt(sum(level * level for level in levels) / len(levels))


def _state_hashes(states: UInt32[Array, "..."]) -> UInt32[Array, "..."]:
    parameters = _hash_parameters()
    mixed = states * jnp.uint32(parameters.multiplier) + jnp.uint32(parameters.increment)
    mixed = mixed ^ (mixed >> jnp.uint32(16))
    mixed = mixed * jnp.uint32(0x85EBCA6B)
    return mixed ^ (mixed >> jnp.uint32(16))


def _states_to_levels(states: UInt32[Array, "..."]) -> Int8[Array, "... 4"]:
    byte_shifts = jnp.arange(_WEIGHTS_PER_STATE, dtype=jnp.uint32) * jnp.uint32(8)
    byte_values = (_state_hashes(states)[..., None] >> byte_shifts) & jnp.uint32(255)
    pair_shifts = jnp.arange(0, 8, 2, dtype=jnp.uint32)
    pairs = jnp.sum((byte_values[..., None] >> pair_shifts) & jnp.uint32(3), axis=-1, dtype=jnp.uint32)
    dither = (jnp.uint32(3) * (byte_values & jnp.uint32(15))) & jnp.uint32(15)
    return (jnp.uint32(8) * pairs + dither).astype(jnp.int32).astype(jnp.int8) - jnp.int8(54)


def _states_to_codewords(states: UInt32[Array, "... steps"]) -> Float[Array, "... cols"]:
    levels = rearrange(_states_to_levels(states), "... steps weights -> ... (steps weights)")
    return levels.astype(jnp.float32) * jnp.float32(_codebook_scale())


def _codebook(window_bits: int) -> Float[Array, "states 4"]:
    states = jnp.arange(1 << window_bits, dtype=jnp.uint32)
    return _states_to_levels(states).astype(jnp.float32) * jnp.float32(_codebook_scale())


@dataclass(frozen=True)
class _TapeLayout:
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
        return ceil(self.blocks * self.spec.block_bits / 32) + 1

    @property
    def state_mask(self) -> int:
        return (1 << self.spec.window_bits) - 1

    @property
    def code_mask(self) -> int:
        return (1 << self.spec.code_bits) - 1

    def window_bit_offsets(self) -> UInt32[np.ndarray, " steps"]:
        block_starts = np.arange(self.blocks) * self.spec.block_bits
        step_offsets = (self.spec.steps_per_block - 1 - np.arange(self.spec.steps_per_block)) * self.spec.code_bits
        return (block_starts[:, None] + step_offsets[None, :]).reshape(-1).astype(np.uint32)

    def field_bit_masks(self) -> UInt32[np.ndarray, " steps"]:
        is_header = np.arange(self.steps) % self.spec.steps_per_block == 0
        return np.where(is_header, self.state_mask, self.code_mask).astype(np.uint32)


def _bytes_to_words(packed_tape: UInt8[Array, "... packed_cols"], word_count: int) -> UInt32[Array, "... words"]:
    *_, row_bytes = packed_tape.shape
    padding = [(0, 0)] * (packed_tape.ndim - 1) + [(0, 4 * word_count - row_bytes)]
    grouped = rearrange(
        jnp.pad(packed_tape, padding).astype(jnp.uint32), "... (words bytes) -> ... words bytes", bytes=4
    )
    byte_shifts = jnp.arange(4, dtype=jnp.uint32) * jnp.uint32(8)
    return jnp.sum(grouped << byte_shifts, axis=-1, dtype=jnp.uint32)


def _words_to_bytes(words: UInt32[Array, "... words"], row_bytes: int) -> UInt8[Array, "... packed_cols"]:
    byte_shifts = jnp.arange(4, dtype=jnp.uint32) * jnp.uint32(8)
    grouped = ((words[..., None] >> byte_shifts) & jnp.uint32(255)).astype(jnp.uint8)
    return rearrange(grouped, "... words bytes -> ... (words bytes)")[..., :row_bytes]


def _tape_to_states(packed_tape: UInt8[Array, "... packed_cols"], layout: _TapeLayout) -> UInt32[Array, "... steps"]:
    words = _bytes_to_words(packed_tape, layout.word_count)
    offsets = layout.window_bit_offsets()
    word_indices = jnp.asarray(offsets // 32, dtype=jnp.int32)
    shifts = jnp.asarray(offsets % 32, dtype=jnp.uint32)
    low = words[..., word_indices] >> shifts
    high = jnp.where(shifts == 0, jnp.uint32(0), words[..., word_indices + 1] << (jnp.uint32(32) - shifts))
    return (low | high) & jnp.uint32(layout.state_mask)


def _states_to_tape(states: UInt32[Array, "... steps"], layout: _TapeLayout) -> UInt8[Array, "... packed_cols"]:
    *leading_dims, _ = states.shape
    offsets = layout.window_bit_offsets()
    word_indices = jnp.asarray(offsets // 32, dtype=jnp.int32)
    shifts = jnp.asarray(offsets % 32, dtype=jnp.uint32)
    values = states & jnp.asarray(layout.field_bit_masks())
    low = values << shifts
    high = jnp.where(shifts == 0, jnp.uint32(0), values >> (jnp.uint32(32) - shifts))
    words = jnp.zeros((*leading_dims, layout.word_count), dtype=jnp.uint32)
    words = words.at[..., word_indices].add(low).at[..., word_indices + 1].add(high)
    return _words_to_bytes(words, layout.row_bytes)


def _backpointer_dtype(code_bits: int) -> DTypeLike:
    if code_bits <= 8:
        return jnp.uint8
    return jnp.uint16


def _segment_states(
    targets: Float[Array, "steps 4"],
    *,
    codebook: Float[Array, "states 4"],
    window_bits: int,
    code_bits: int,
) -> UInt32[Array, " steps"]:
    num_predecessors = 1 << code_bits
    num_suffixes = 1 << (window_bits - code_bits)
    codeword_norms = jnp.sum(jnp.square(codebook), axis=-1)
    backpointer_dtype = _backpointer_dtype(code_bits)

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
        best_predecessors = jnp.argmin(grouped, axis=0).astype(backpointer_dtype)
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


def _weights_to_states(weights: Float[Array, "... cols"], layout: _TapeLayout) -> UInt32[Array, "... steps"]:
    *leading_dims, cols = weights.shape
    segments = rearrange(
        weights.reshape(-1, cols),
        "rows (blocks steps weights) -> (rows blocks) steps weights",
        steps=layout.spec.steps_per_block,
        weights=_WEIGHTS_PER_STATE,
    )
    codebook = _codebook(layout.spec.window_bits)
    num_states, _ = codebook.shape
    backpointers_per_segment = (num_states >> layout.spec.code_bits) * layout.spec.steps_per_block
    segments_per_chunk = max(
        1,
        min(_MAX_STATES_PER_CHUNK // num_states, _MAX_BACKPOINTERS_PER_CHUNK // backpointers_per_segment),
    )
    fit_segment = partial(
        _segment_states,
        codebook=codebook,
        window_bits=layout.spec.window_bits,
        code_bits=layout.spec.code_bits,
    )
    states = jax.lax.map(fit_segment, segments, batch_size=segments_per_chunk)
    return states.reshape(*leading_dims, layout.steps)


def _search_scales(weights: Float[Array, "... cols"]) -> Float[Array, "..."]:
    scales = jnp.sqrt(jnp.mean(jnp.square(weights), axis=-1))
    return jnp.where(scales == 0, 1, scales)


def _least_squares_scales(
    weights: Float[Array, "... cols"],
    codewords: Float[Array, "... cols"],
) -> Float[Array, "..."]:
    correlations = jnp.sum(weights * codewords, axis=-1)
    energies = jnp.sum(jnp.square(codewords), axis=-1)
    safe_energies = jnp.where(energies == 0, 1, energies)
    return jnp.where(energies == 0, 0, correlations / safe_energies)


class _PackedParameters(NamedTuple):
    packed_tape: UInt8[Array, "... packed_cols"]
    scales: Float[Array, "..."]


@supports_dummy_arrays()
def _weights_to_packed_parameters(
    weights: Float[Array, "... cols"],
    layout: _TapeLayout,
    *,
    sharding_config: ShardingConfig,
) -> _PackedParameters:
    scratch_sharding = sharding_config.make_sharding((None,) * weights.ndim)
    targets = with_sharding(weights, scratch_sharding).astype(jnp.float32)
    search_scales = _search_scales(targets)
    states = _weights_to_states(targets / search_scales[..., None], layout)
    scales = _least_squares_scales(targets, _states_to_codewords(states))
    return _PackedParameters(_states_to_tape(states, layout), scales.astype(weights.dtype))


@supports_dummy_arrays(out_sharding_rule=preserve_first_input_sharding)
def _packed_parameters_to_weights(
    packed_tape: UInt8[Array, "... packed_cols"],
    scales: Float[Array, "..."],
    layout: _TapeLayout,
    dtype: DTypeLike,
) -> Float[Array, "... cols"]:
    levels = _states_to_levels(_tape_to_states(packed_tape, layout))
    levels = rearrange(levels, "... steps weights -> ... (steps weights)").astype(jnp.float32)
    combined_scales = scales.astype(jnp.float32) * jnp.float32(_codebook_scale())
    return (levels * combined_scales[..., None]).astype(dtype)


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
        if self.restart_columns <= 0 or self.restart_columns % _WEIGHTS_PER_STATE != 0:
            raise ValueError(
                f"restart_columns must be a positive multiple of {_WEIGHTS_PER_STATE}, got {self.restart_columns}"
            )

    @property
    def code_bits(self) -> int:
        return self.bits * _WEIGHTS_PER_STATE

    @property
    def steps_per_block(self) -> int:
        return self.restart_columns // _WEIGHTS_PER_STATE

    @property
    def block_bits(self) -> int:
        return self.window_bits + (self.steps_per_block - 1) * self.code_bits

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
        if preconditioner is not None:
            raise ValueError("Trellis compression does not support preconditioning.")

        weight_axes = self.layout.weight_partition(weights.ndim - 2, is_sharded=is_sharded)
        *parameter_axes, _ = weight_axes
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        stored_weights = self.layout.from_output_input(weights, sharding=weight_sharding)
        *leading_dims, cols = stored_weights.shape
        layout = _TapeLayout(self, cols)

        if is_dummy_array(weights):
            packed_tape = dummy_array(
                (*leading_dims, layout.row_bytes),
                jnp.uint8,
                sharding_config.resolve_sharding((*parameter_axes, None)),
            )
            scales = dummy_array(tuple(leading_dims), weights.dtype, sharding_config.resolve_sharding(parameter_axes))
        else:
            if self.window_bits > _MAX_SEARCH_WINDOW_BITS:
                raise ValueError(
                    f"Searching 2**{self.window_bits} states is not supported, "
                    f"compression needs window_bits of at most {_MAX_SEARCH_WINDOW_BITS}"
                )
            packed_tape, scales = _weights_to_packed_parameters(
                stored_weights, layout, sharding_config=sharding_config
            )

        return self.from_packed_parameters(
            packed_tape=packed_tape,
            scales=scales,
            cols=cols,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
        )

    def from_packed_parameters(
        self,
        *,
        packed_tape: UInt8[Array, "*components rows packed_cols"],
        scales: Float[Array, "*components rows"],
        cols: int,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "TrellisMatrix":
        layout = _TapeLayout(self, cols)
        *_, row_bytes = packed_tape.shape
        if row_bytes != layout.row_bytes:
            raise ValueError(f"Expected {layout.row_bytes}-byte tape rows for cols={cols}, got {row_bytes}")

        *parameter_axes, _cols_axis = self.layout.weight_partition(scales.ndim - 1, is_sharded=is_sharded)
        return TrellisMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            cols=cols,
            packed_tape=with_sharding(packed_tape, sharding_config.resolve_sharding((*parameter_axes, None))),
            scales=with_sharding(scales, sharding_config.resolve_sharding(tuple(parameter_axes))),
        )


class TrellisMatrix(EmbeddingMatrix[TrellisSpec]):
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

    def switch_sharding_config(self, sharding_config: ShardingConfig) -> "TrellisMatrix":
        return self.spec.from_packed_parameters(
            packed_tape=self.packed_tape,
            scales=self.scales,
            cols=self.cols,
            sharding_config=sharding_config,
            is_sharded=self.is_sharded,
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
        return TrellisMatrix(
            spec=self.spec,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
            cols=self.cols,
            packed_tape=load_as(self.packed_tape, exported_data.arrays[prefix / "weights"]),
            scales=load_as(self.scales, exported_data.arrays[prefix / "scales"]),
        )

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        return self.spec.layout.to_output_input(self._weights_for_forward(self.dtype))

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        dtype: DTypeLike | None = None,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "*batch out_channels"]:
        self._raise_if_batched()
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError(f"Embedding lookup not supported for layout {self.spec.layout}")
        if dtype is None:
            dtype = self.dtype
        return self._weights_for_forward(dtype, row_index)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        self._raise_if_batched()
        weights = self._weights_for_forward(vector.dtype)
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(weights, vector)

    def _weights_for_forward(
        self,
        dtype: DTypeLike,
        row_index: int | Int[Array, "*batch"] | None = None,
    ) -> Float[Array, "... cols"]:
        layout = _TapeLayout(self.spec, self.cols)
        if row_index is not None:
            packed_tape = lookup_sharded_indices(self.packed_tape, row_index)
            scales = lookup_sharded_indices(self.scales, row_index)
            return _packed_parameters_to_weights(packed_tape, scales, layout, dtype)
        weights = _packed_parameters_to_weights(self.packed_tape, self.scales, layout, dtype)
        weight_axes = self.spec.layout.weight_partition(self.scales.ndim - 1, is_sharded=self.is_sharded)
        return with_sharding(weights, self._resolve_sharding(weight_axes))
